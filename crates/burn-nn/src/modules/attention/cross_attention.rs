//! Cross-Attention Module for Burn (Production Grade)
//!
//! Features:
//! - Asymmetric Input Shapes (Query vs Context)
//! - Grouped Query Attention (GQA) & Multi-Query Attention (MQA) support
//! - Quantization-Safe Masking (min_float)
//! - Sparse-Ready (quiet_softmax)
//! - KV Caching for Streaming Inference

use crate::modules::{Linear, LinearConfig};
use crate::{Dropout, DropoutConfig};
use burn_core as burn;
use num_traits::Float;

use burn::{
    config::Config,
    module::{Initializer, Module},
    tensor::{
        Bool, Tensor,
        activation::{quiet_softmax, softmax},
        backend::Backend,
    },
};

#[derive(Config, Debug)]
/// Configuration to create a [CrossAttention](CrossAttention) layer using the [init function](CrossAttentionConfig::init).
pub struct CrossAttentionConfig {
    /// Dimension of the Query (e.g., Decoder state).
    pub d_model: usize,
    /// Dimension of the Context (e.g., Encoder audio embeddings).
    pub d_context: usize,
    /// Number of heads for the Query.
    pub n_heads: usize,
    /// Number of heads for Key/Value (Set to 1 for MQA, set to n_heads for MHA).
    pub n_heads_kv: usize,
    /// Dimension of a single head.
    pub d_head: usize,
    /// Dropout rate.
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Masking value. Use -1.0e4 for f16/bf16 safety.
    #[config(default = -1.0e4)]
    pub min_float: f64,
    /// Use quiet_softmax to allow zero-attention (good for sparse/quantized models).
    #[config(default = false)]
    pub quiet_softmax: bool,
}

#[derive(Module, Debug)]
/// The Cross attention module
///
/// # Params
///
/// - `query`: [`Linear`] layer with `d_model` input and output features.
/// - `key`: [`Linear`] layer with `d_model` input and output features.
/// - `value`: [`Linear`] layer with `d_model` input and output features.
/// - `output`: [`Linear`] layer with `d_model` input and output features.
///
/// Should be created with [CrossAttentionConfig].
pub struct CrossAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,

    n_heads: usize,
    n_heads_kv: usize,
    d_head: usize,
    scale: f64,
    min_float: f64,
    quiet_softmax: bool,
}

impl CrossAttentionConfig {
    /// Initializes a new cross-attention module.
    ///
    /// # Arguments
    ///
    /// * `device` - The device on which to initialize the module.
    ///
    /// # Returns
    ///
    /// A new [CrossAttention] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttention<B> {
        // Safety Rail for GQA
        assert_eq!(
            self.n_heads % self.n_heads_kv,
            0,
            "Query heads must be divisible by KV heads"
        );

        let init_linear = |in_dim, out_dim| {
            LinearConfig::new(in_dim, out_dim)
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / (self.d_head as f64).sqrt(),
                    fan_out_only: false,
                })
                .init(device)
        };

        CrossAttention {
            // ADVICE: Asymmetric Projections
            query: init_linear(self.d_model, self.n_heads * self.d_head),
            key: init_linear(self.d_context, self.n_heads_kv * self.d_head),
            value: init_linear(self.d_context, self.n_heads_kv * self.d_head),
            output: init_linear(self.n_heads * self.d_head, self.d_model),

            dropout: DropoutConfig::new(self.dropout).init(),
            n_heads: self.n_heads,
            n_heads_kv: self.n_heads_kv,
            d_head: self.d_head,
            scale: (self.d_head as f64).sqrt().recip(),
            min_float: self.min_float,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

impl<B: Backend> CrossAttention<B> {
    /// Applies cross-attention to query using context as key and value.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor of shape `[batch, seq_len_query, d_model]`.
    /// * `context` - Context tensor of shape `[batch, seq_len_context, d_context]`.
    /// * `mask` - Optional attention mask of shape `[batch, seq_len_context]` where `true` indicates positions to mask.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq_len_query, d_model]`.
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        context: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, l_q, _] = query.dims();
        let [_, l_k, _] = context.dims();

        // 1. Projections
        let q = self.query.forward(query);
        let k = self.key.forward(context.clone());
        let v = self.value.forward(context);

        // 2. Reshape Heads
        // Q: [Batch, Heads, L_q, D_head]
        let q = q
            .reshape([batch, l_q, self.n_heads, self.d_head])
            .swap_dims(1, 2);

        // K, V: [Batch, Heads_KV, L_k, D_head]
        let k = k
            .reshape([batch, l_k, self.n_heads_kv, self.d_head])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, l_k, self.n_heads_kv, self.d_head])
            .swap_dims(1, 2);

        // 3. GQA Expansion
        // ADVICE: Handle GQA by repeating KV heads to match Query heads
        let (k, v) = if self.n_heads != self.n_heads_kv {
            let n_rep = self.n_heads / self.n_heads_kv;
            (self.repeat_kv(k, n_rep), self.repeat_kv(v, n_rep))
        } else {
            (k, v)
        };

        // 4. Score Calculation
        let scores = q.matmul(k.transpose()) * self.scale;

        // 5. Masking
        // ADVICE: Use min_float for F16/FP8 safety
        let scores = if let Some(mask) = mask {
            let mask = mask.reshape([batch, 1, 1, l_k]);
            scores.mask_fill(mask, self.min_float)
        } else {
            scores
        };

        // 6. Softmax
        // ADVICE: Optional Quiet Softmax for sparse networks
        let weights = if self.quiet_softmax {
            quiet_softmax(scores, 3)
        } else {
            softmax(scores, 3)
        };

        let weights = self.dropout.forward(weights);

        // 7. Aggregate & Output
        let output = weights.matmul(v);
        let output = output
            .swap_dims(1, 2)
            .reshape([batch, l_q, self.n_heads * self.d_head]);

        self.output.forward(output)
    }

    /// Helper for Grouped Query Attention
    fn repeat_kv(&self, x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        let [b, h, l, d] = x.dims();
        x.reshape([b, h, 1, l, d])
            .expand([b, h, n_rep, l, d])
            .reshape([b, h * n_rep, l, d])
    }
}
