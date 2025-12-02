//! Cross-Attention Module for Burn
//!
//! Features:
//! - Asymmetric Input Shapes (Query vs Context)
//! - Grouped Query Attention (GQA) & Multi-Query Attention (MQA) support
//! - Quantization-Safe Masking (min_float)
//! - Sparse-Ready (quiet_softmax)
//! - KV Caching for Streaming Inference

use crate::cache::TensorCache;
use crate::modules::{Linear, LinearConfig};
use crate::{Dropout, DropoutConfig};
use burn_core as burn;

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

/// Cache for the [Cross Attention](CrossAttention) layer.
///
/// To be used during inference when context is constant.
pub struct CrossAttentionCache<B: Backend> {
    /// Cached key tensor.
    pub k: TensorCache<B, 4>,
    /// Cached value tensor.
    pub v: TensorCache<B, 4>,
}

impl<B: Backend> CrossAttentionCache<B> {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            k: TensorCache::empty(),
            v: TensorCache::empty(),
        }
    }
}

impl<B: Backend> Default for CrossAttentionCache<B> {
    fn default() -> Self {
        Self::new()
    }
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

    /// Applies cross-attention to query using context as key and value.
    ///
    /// This method uses a cache to avoid recomputing key and value tensors when the context is the same.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor of shape `[batch, seq_len_query, d_model]`.
    /// * `context` - Context tensor of shape `[batch, seq_len_context, d_context]`.
    /// * `mask` - Optional attention mask of shape `[batch, seq_len_context]` where `true` indicates positions to mask.
    /// * `cache` - The cache to use.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq_len_query, d_model]`.
    pub fn forward_cache(
        &self,
        query: Tensor<B, 3>,
        context: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
        cache: &mut CrossAttentionCache<B>,
    ) -> Tensor<B, 3> {
        let [batch, l_q, _] = query.dims();

        // 1. Projections
        let q = self.query.forward(query);

        let k_compute = |context: Tensor<B, 3>| {
            let [batch, l_k, _] = context.dims();
            self.key
                .forward(context)
                .reshape([batch, l_k, self.n_heads_kv, self.d_head])
                .swap_dims(1, 2)
        };
        let v_compute = |context: Tensor<B, 3>| {
            let [batch, l_k, _] = context.dims();
            self.value
                .forward(context)
                .reshape([batch, l_k, self.n_heads_kv, self.d_head])
                .swap_dims(1, 2)
        };

        let k = cache.k.forward_full(context.clone(), k_compute);
        let v = cache.v.forward_full(context, v_compute);

        let [_, _, l_k, _] = k.dims();

        // 2. Reshape Heads
        // Q: [Batch, Heads, L_q, D_head]
        let q = q
            .reshape([batch, l_q, self.n_heads, self.d_head])
            .swap_dims(1, 2);

        // K, V are already in their correct shape from k_compute and v_compute

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{Distribution, Int, Shape, Tensor, Tolerance};

    #[test]
    fn test_cross_attention_mha_shapes() {
        let [
            batch_size,
            seq_len_query,
            seq_len_context,
            d_model,
            d_context,
            n_heads,
            d_head,
        ] = [7, 13, 15, 32, 40, 4, 8];
        let device = Default::default();
        let config = CrossAttentionConfig {
            d_model,
            d_context,
            n_heads,
            n_heads_kv: n_heads, // MHA case
            d_head,
            dropout: 0.1,
            min_float: -1.0e4,
            quiet_softmax: false,
        };
        let cross_attn = config.init::<TestBackend>(&device);

        let query = Tensor::random(
            [batch_size, seq_len_query, d_model],
            Distribution::Default,
            &device,
        );
        let context = Tensor::random(
            [batch_size, seq_len_context, d_context],
            Distribution::Default,
            &device,
        );

        let output = cross_attn.forward(query, context, None);

        assert_eq!(
            output.shape(),
            Shape::new([batch_size, seq_len_query, d_model]),
            "Output should have the correct shape",
        );
    }

    #[test]
    fn test_cross_attention_gqa_shapes() {
        let [
            batch_size,
            seq_len_query,
            seq_len_context,
            d_model,
            d_context,
            n_heads,
            n_heads_kv,
            d_head,
        ] = [7, 13, 15, 32, 40, 4, 2, 8];
        let device = Default::default();
        let config = CrossAttentionConfig {
            d_model,
            d_context,
            n_heads,
            n_heads_kv, // GQA case
            d_head,
            dropout: 0.1,
            min_float: -1.0e4,
            quiet_softmax: false,
        };
        let cross_attn = config.init::<TestBackend>(&device);

        let query = Tensor::random(
            [batch_size, seq_len_query, d_model],
            Distribution::Default,
            &device,
        );
        let context = Tensor::random(
            [batch_size, seq_len_context, d_context],
            Distribution::Default,
            &device,
        );

        let output = cross_attn.forward(query, context, None);

        assert_eq!(
            output.shape(),
            Shape::new([batch_size, seq_len_query, d_model]),
            "Output should have the correct shape",
        );
    }

    #[test]
    fn test_cross_attention_mqa_shapes() {
        let [
            batch_size,
            seq_len_query,
            seq_len_context,
            d_model,
            d_context,
            n_heads,
            d_head,
        ] = [7, 13, 15, 32, 40, 4, 8];
        let device = Default::default();
        let config = CrossAttentionConfig {
            d_model,
            d_context,
            n_heads,
            n_heads_kv: 1, // MQA case
            d_head,
            dropout: 0.1,
            min_float: -1.0e4,
            quiet_softmax: false,
        };
        let cross_attn = config.init::<TestBackend>(&device);

        let query = Tensor::random(
            [batch_size, seq_len_query, d_model],
            Distribution::Default,
            &device,
        );
        let context = Tensor::random(
            [batch_size, seq_len_context, d_context],
            Distribution::Default,
            &device,
        );

        let output = cross_attn.forward(query, context, None);

        assert_eq!(
            output.shape(),
            Shape::new([batch_size, seq_len_query, d_model]),
            "Output should have the correct shape",
        );
    }

    #[test]
    fn test_cross_attention_mask() {
        let [
            batch_size,
            seq_len_query,
            seq_len_context,
            d_model,
            d_context,
            n_heads,
            d_head,
        ] = [3, 6, 8, 12, 16, 4, 3];
        let num_padded = 2;
        let device = Default::default();
        let config = CrossAttentionConfig {
            d_model,
            d_context,
            n_heads,
            n_heads_kv: n_heads,
            d_head,
            dropout: 0.0, // No dropout for deterministic test
            min_float: -1.0e4,
            quiet_softmax: false,
        };
        let cross_attn = config.init::<TestBackend>(&device);

        // Create a padding mask for the context
        let mut mask: Tensor<TestBackend, 2, Int> =
            Tensor::zeros([batch_size, seq_len_context], &device);
        mask = mask.slice_assign(
            [0..batch_size, seq_len_context - num_padded..seq_len_context],
            Tensor::ones([batch_size, num_padded], &device),
        );
        let mask_bool = mask.equal_elem(1);

        let query = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len_query, d_model],
            Distribution::Default,
            &device,
        );

        let context_1 = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len_context, d_context],
            Distribution::Default,
            &device,
        );

        // Change the padded part of the context tensor
        let context_2 = context_1.clone().slice_assign(
            [
                0..batch_size,
                seq_len_context - num_padded..seq_len_context,
                0..d_context,
            ],
            Tensor::random(
                [batch_size, num_padded, d_context],
                Distribution::Default,
                &device,
            ),
        );

        // The outputs should be the same since the changed part is masked.
        let output_1 = cross_attn.forward(query.clone(), context_1, Some(mask_bool.clone()));
        let output_2 = cross_attn.forward(query, context_2, Some(mask_bool));

        output_1
            .into_data()
            .assert_approx_eq(&output_2.into_data(), Tolerance::<f32>::default());
    }

    #[test]
    #[should_panic]
    fn test_gqa_panic_if_n_heads_not_divisible_by_n_heads_kv() {
        let device = Default::default();
        let config = CrossAttentionConfig {
            d_model: 32,
            d_context: 32,
            n_heads: 5,
            n_heads_kv: 2,
            d_head: 8,
            dropout: 0.1,
            min_float: -1.0e4,
            quiet_softmax: false,
        };
        config.init::<TestBackend>(&device);
    }

    #[test]
    fn test_cross_attention_cache() {
        let [
            batch_size,
            seq_len_query,
            seq_len_context,
            d_model,
            d_context,
            n_heads,
            d_head,
        ] = [3, 6, 8, 12, 16, 4, 3];
        let device = Default::default();
        let config = CrossAttentionConfig {
            d_model,
            d_context,
            n_heads,
            n_heads_kv: n_heads,
            d_head,
            dropout: 0.0, // No dropout for deterministic test
            min_float: -1.0e4,
            quiet_softmax: false,
        };
        let cross_attn = config.init::<TestBackend>(&device);

        let query1 = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len_query, d_model],
            Distribution::Default,
            &device,
        );
        let context = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len_context, d_context],
            Distribution::Default,
            &device,
        );

        // First forward pass, no cache
        let output1 = cross_attn.forward(query1.clone(), context.clone(), None);

        // Second forward pass with cache
        let mut cache = CrossAttentionCache::new();
        let output2 = cross_attn.forward_cache(query1.clone(), context.clone(), None, &mut cache);

        // The two outputs should be identical
        output1
            .into_data()
            .assert_approx_eq(&output2.into_data(), Tolerance::<f32>::default());

        // Third forward pass with different query, but same context and cache
        let query2 = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len_query, d_model],
            Distribution::Default,
            &device,
        );
        let output3 = cross_attn.forward_cache(query2.clone(), context.clone(), None, &mut cache);

        // For control, do a forward pass without cache with query2
        let output4 = cross_attn.forward(query2.clone(), context.clone(), None);

        // output3 and output4 should be identical
        output3
            .into_data()
            .assert_approx_eq(&output4.into_data(), Tolerance::<f32>::default());
    }
}
