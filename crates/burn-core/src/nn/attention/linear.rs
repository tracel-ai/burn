use crate as burn;
use burn::serde::{Deserialize, Serialize};

use crate::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay};
use crate::nn::{Initializer, Linear, LinearConfig};
use crate::{
    config::Config,
    tensor::{Bool, Tensor, activation, backend::Backend},
};

/// Configuration to create a [LinearAttention] layer using the [init function](LinearAttentionConfig::init).
#[derive(Config, Debug)]
pub struct LinearAttentionConfig {
    /// The size of each linear layer.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// Kernel type to ensure positive feature maps.
    #[config(default = "KernelType::Relu")]
    pub kernel: KernelType,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Kernel type for linear attention (positive feature maps).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KernelType {
    /// phi(x) = relu(x)
    Relu,
}

/// Input for [LinearAttention].
#[derive(Debug, Clone)]
pub struct LinearAttnInput<B: Backend> {
    /// Shape `[batch_size, seq_length_1, d_model]`
    pub query: Tensor<B, 3>,
    /// Shape `[batch_size, seq_length_2, d_model]`
    pub key: Tensor<B, 3>,
    /// Shape `[batch_size, seq_length_2, d_model]`
    pub value: Tensor<B, 3>,
    /// Optional padding mask for key/value: shape `[batch_size, seq_length_2]` (true = masked)
    pub mask_pad: Option<Tensor<B, 2, Bool>>,
}

impl<B: Backend> LinearAttnInput<B> {
    /// Create a self-attention input with the same tensor.
    pub fn self_attn(tensor: Tensor<B, 3>) -> Self {
        Self { query: tensor.clone(), key: tensor.clone(), value: tensor, mask_pad: None }
    }

    /// Attach a padding mask.
    pub fn mask_pad(mut self, mask: Tensor<B, 2, Bool>) -> Self {
        self.mask_pad = Some(mask);
        self
    }
}

/// Linear attention module using kernel feature maps for Q and K.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LinearAttention<B: Backend> {
    /// Query projection.
    pub query: Linear<B>,
    /// Key projection.
    pub key: Linear<B>,
    /// Value projection.
    pub value: Linear<B>,
    /// Output projection.
    pub output: Linear<B>,
    /// Model dimension.
    pub d_model: usize,
    /// Number of heads.
    pub n_heads: usize,
    /// Head dimension.
    pub d_k: usize,
    /// Kernel mapping used for positive features (ignored for recording).
    pub kernel: Ignored<KernelType>,
}

impl<B: Backend> ModuleDisplay for LinearAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_heads", &self.n_heads)
            .add("d_k", &self.d_k)
            .optional()
    }
}

impl LinearAttentionConfig {
    /// Initialize a new [LinearAttention] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearAttention<B> {
        let linear = |in_features, out_features| {
            LinearConfig::new(in_features, out_features)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        assert!(
            self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads"
        );
        let d_k = self.d_model / self.n_heads;

        LinearAttention {
            query: linear(self.d_model, self.d_model),
            key: linear(self.d_model, self.d_model),
            value: linear(self.d_model, self.d_model),
            output: linear(self.d_model, self.d_model),
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_k,
            kernel: Ignored(self.kernel),
        }
    }
}

/// Output of [LinearAttention].
#[derive(Debug, Clone)]
pub struct LinearAttnOutput<B: Backend> {
    /// Context tensor of shape `[batch_size, seq_length_1, d_model]`.
    pub context: Tensor<B, 3>,
}

impl<B: Backend> LinearAttention<B> {
    /// Forward pass.
    pub fn forward(&self, input: LinearAttnInput<B>) -> LinearAttnOutput<B> {
        let [batch_size, seq_length_q, _] = input.query.dims();

        // Project and reshape to [B, nH, S, d_k]
        let q = self.attn_linear(input.query, &self.query);
        let mut k = self.attn_linear(input.key, &self.key);
        let mut v = self.attn_linear(input.value, &self.value);

        // Apply padding mask on keys/values if provided.
        if let Some(mask_pad) = input.mask_pad {
            // mask_pad: [B, S2] (true = masked). Broadcast to [B, 1, S2, 1]
            let mask = mask_pad.clone().reshape([batch_size, 1, mask_pad.dims()[1], 1]);
            // Zero-out masked positions.
            k = k.mask_fill(mask.clone(), 0.0);
            v = v.mask_fill(mask, 0.0);
        }

        // Kernel feature maps (positive).
        let q_phi = match self.kernel.0 {
            KernelType::Relu => activation::relu(q),
        };
        let k_phi = match self.kernel.0 {
            KernelType::Relu => activation::relu(k),
        };

        // Compute KV = (K_phi^T V) over the sequence dimension -> [B,nH,d_k,d_k]
        let kv = k_phi.clone().swap_dims(2, 3).matmul(v);

        // Compute normalizer: denom = q_phi @ (sum_i k_phi[i]) -> [B,nH,Sq,1]
        // k_phi.sum_dim(2): [B,nH,1,d_k] -> swap to [B,nH,d_k,1]
        let k_sum = k_phi.sum_dim(2).swap_dims(2, 3);
        let denom = q_phi.clone().matmul(k_sum).add_scalar(1e-6);

        // Numerator: q_phi @ KV -> [B,nH,Sq,d_k]
        let context = q_phi.matmul(kv);

        // Normalize
        let context = context / denom;

        // Back to [B,Sq,d_model]
        let context = context.swap_dims(1, 2).reshape([batch_size, seq_length_q, self.d_model]);
        let context = self.output.forward(context);

        LinearAttnOutput { context }
    }

    fn attn_linear(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}
