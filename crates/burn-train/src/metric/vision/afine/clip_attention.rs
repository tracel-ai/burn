//! Self-attention block matching PyTorch's `nn.MultiheadAttention` wire
//! format.
//!
//! PyTorch stores Q/K/V as a single fused `in_proj_weight` of shape
//! `[3 * d_model, d_model]` and `in_proj_bias` of shape `[3 * d_model]`.
//! Burn's [`burn_nn::attention::MultiHeadAttention`] uses three separate
//! Linear layers, so the CLIP checkpoint cannot map to it without
//! pre-splitting the weights at load time.
//!
//! This module keeps the fused layout: a single `qkv_proj` Linear of
//! shape `(d_model -> 3 * d_model)` and a `chunk(3, -1)` at forward,
//! giving a one-to-one mapping with the checkpoint.
//!
//! No attention mask is supported. CLIP's image encoder runs
//! unconditional self-attention; the text encoder (which uses a causal
//! mask) is not ported.

use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn_nn::{Linear, LinearConfig};

/// Configuration for [`ClipQkvAttention`].
#[derive(Config, Debug)]
pub(crate) struct ClipQkvAttentionConfig {
    /// Embedding dimension. Must be divisible by `n_heads`.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

impl ClipQkvAttentionConfig {
    /// Initialize a [`ClipQkvAttention`] block.
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> ClipQkvAttention<B> {
        assert_eq!(
            self.d_model % self.n_heads,
            0,
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model,
            self.n_heads
        );
        let head_dim = self.d_model / self.n_heads;
        ClipQkvAttention {
            qkv_proj: LinearConfig::new(self.d_model, 3 * self.d_model)
                .with_bias(true)
                .init(device),
            out_proj: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(true)
                .init(device),
            d_model: self.d_model,
            n_heads: self.n_heads,
            head_dim,
        }
    }
}

/// Self-attention with a fused QKV projection, matching CLIP's checkpoint
/// layout one-to-one.
#[derive(Module, Debug)]
pub(crate) struct ClipQkvAttention<B: Backend> {
    /// Fused projection `d_model -> 3 * d_model` for Q, K, V.
    pub(crate) qkv_proj: Linear<B>,
    /// Output projection `d_model -> d_model`.
    pub(crate) out_proj: Linear<B>,
    /// Embedding dimension.
    pub(crate) d_model: usize,
    /// Number of attention heads.
    pub(crate) n_heads: usize,
    /// Per-head dimension (`d_model / n_heads`).
    pub(crate) head_dim: usize,
}

impl<B: Backend> ClipQkvAttention<B> {
    /// Apply self-attention. Input and output shape: `[batch, seq, d_model]`.
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();

        let qkv = self.qkv_proj.forward(x);
        let mut chunks = qkv.chunk(3, 2);
        let value = chunks.remove(2);
        let key = chunks.remove(1);
        let query = chunks.remove(0);

        let to_heads = |t: Tensor<B, 3>| {
            t.reshape([batch, seq, self.n_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let query = to_heads(query);
        let key = to_heads(key);
        let value = to_heads(value);

        let scale = (self.head_dim as f32).sqrt();
        let scores = query.matmul(key.transpose()).div_scalar(scale);
        let weights = softmax(scores, 3);

        let context = weights
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([batch, seq, self.d_model]);
        self.out_proj.forward(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_flex::Flex;

    type TestBackend = Flex;

    #[test]
    fn clip_qkv_attention_preserves_shape() {
        let device = Default::default();
        let attn = ClipQkvAttentionConfig::new(768, 12).init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([1, 50, 768], Distribution::Default, &device);
        let output = attn.forward(input);

        assert_eq!(output.dims(), [1, 50, 768]);
    }

    #[test]
    fn clip_qkv_attention_handles_batch() {
        let device = Default::default();
        let attn = ClipQkvAttentionConfig::new(64, 4).init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([3, 16, 64], Distribution::Default, &device);
        let output = attn.forward(input);

        assert_eq!(output.dims(), [3, 16, 64]);
    }
}
