//! CLIP ViT-B/32 image encoder.
//!
//! Structurally a port of OpenAI CLIP's `VisionTransformer` minus the
//! text-side and joint-embedding projection. A-FINE consumes the
//! per-layer patch features (twelve `[batch, num_patches, 768]` tensors),
//! not the final CLIP embedding, so [`Self::forward_with_features`] is
//! the entry point used by the metric. [`Self::forward`] returns just the
//! `ln_post`-normalized class-token embedding and is provided for
//! completeness.
//!
//! Layout: NLD (`[batch, seq, embed]`) end-to-end. The PyTorch reference
//! permutes to LND (`[seq, batch, embed]`) inside the transformer stack
//! and back again before `ln_post`; attention is invariant to that swap,
//! so we stay in NLD throughout.

use burn_core as burn;

use burn::config::Config;
use burn::module::{Initializer, Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};

use super::clip_attention::{ClipQkvAttention, ClipQkvAttentionConfig};
use super::quick_gelu::QuickGelu;

/// Configuration for [`ClipVisualEncoder`].
#[derive(Config, Debug)]
pub struct ClipVisualEncoderConfig {
    /// Input image channels. Defaults to 3 (RGB).
    #[config(default = "3")]
    pub in_channels: usize,
    /// Token embedding dimension. Defaults to 768 (CLIP ViT-B/32).
    #[config(default = "768")]
    pub embed_dim: usize,
    /// Patch (and conv stride) size. Defaults to 32 (CLIP ViT-B/32).
    #[config(default = "32")]
    pub patch_size: usize,
    /// Number of transformer blocks. Defaults to 12.
    #[config(default = "12")]
    pub num_layers: usize,
    /// Number of attention heads per block. Defaults to 12.
    #[config(default = "12")]
    pub num_heads: usize,
    /// Hidden dimension of the per-block MLP. Defaults to 3072 (4 * 768).
    #[config(default = "3072")]
    pub mlp_dim: usize,
    /// Expected input resolution. Defaults to 256. Must be a multiple of `patch_size`.
    #[config(default = "256")]
    pub image_size: usize,
}

impl ClipVisualEncoderConfig {
    /// Initialize a [`ClipVisualEncoder`] with random weights.
    pub fn init(&self, device: &Device) -> ClipVisualEncoder {
        assert_eq!(
            self.image_size % self.patch_size,
            0,
            "image_size ({}) must be a multiple of patch_size ({})",
            self.image_size,
            self.patch_size
        );

        let num_patches = (self.image_size / self.patch_size).pow(2);
        let seq_len = num_patches + 1; // +1 for the prepended CLS token.

        let patch_embed = Conv2dConfig::new(
            [self.in_channels, self.embed_dim],
            [self.patch_size, self.patch_size],
        )
        .with_stride([self.patch_size, self.patch_size])
        .with_bias(false)
        .init(device);

        // Standard transformer init magnitude. Final values get overwritten
        // when pretrained weights load; this just keeps random-init forward
        // passes well-conditioned for tests.
        let init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let class_token = init.init([self.embed_dim], device);
        let positional_embedding = init.init([seq_len, self.embed_dim], device);

        let blocks = (0..self.num_layers)
            .map(|_| {
                TransformerBlockConfig::new(self.embed_dim, self.num_heads, self.mlp_dim)
                    .init(device)
            })
            .collect();

        ClipVisualEncoder {
            patch_embed,
            class_token,
            positional_embedding,
            ln_pre: LayerNormConfig::new(self.embed_dim).init(device),
            blocks,
            ln_post: LayerNormConfig::new(self.embed_dim).init(device),
            embed_dim: self.embed_dim,
            patch_size: self.patch_size,
            image_size: self.image_size,
        }
    }
}

/// Output of [`ClipVisualEncoder::forward_with_features`].
///
/// `features` are the per-block patch tokens A-FINE consumes. `cls` is
/// the post-`ln_post` class-token embedding, present only when the
/// caller requests it; the metric path skips the slice + LayerNorm to
/// avoid wasted work.
#[derive(Debug)]
pub struct ClipOutput {
    /// Per-block patch tokens, length `num_layers`, each
    /// `[batch, num_patches, embed_dim]`.
    pub features: Vec<Tensor<3>>,
    /// Post-`ln_post` class-token embedding, `[batch, embed_dim]`. `Some`
    /// when `forward_with_features` is called with `return_cls = true`.
    pub cls: Option<Tensor<2>>,
}

/// CLIP ViT-B/32 image encoder.
///
/// Consumes a normalized RGB image and returns either the class-token
/// embedding ([`Self::forward`]) or the per-layer patch-token features
/// used by A-FINE ([`Self::forward_with_features`]).
#[derive(Module, Debug)]
pub struct ClipVisualEncoder {
    /// Patch-embedding convolution: `(in_channels, embed_dim)`, kernel and
    /// stride both `patch_size`, no bias (matches CLIP).
    pub(crate) patch_embed: Conv2d,
    /// Learnable class token, shape `[embed_dim]`. Stored 1-D to match the
    /// CLIP checkpoint key `class_embedding`; reshaped to `[1, 1, embed]`
    /// at forward time.
    pub(crate) class_token: Param<Tensor<1>>,
    /// Learnable positional embedding, shape `[seq_len, embed_dim]`.
    pub(crate) positional_embedding: Param<Tensor<2>>,
    /// Pre-stack LayerNorm.
    pub(crate) ln_pre: LayerNorm,
    /// Stack of transformer blocks.
    pub(crate) blocks: Vec<TransformerBlock>,
    /// Post-stack LayerNorm applied to the class token only.
    pub(crate) ln_post: LayerNorm,

    pub(crate) embed_dim: usize,
    pub(crate) patch_size: usize,
    pub(crate) image_size: usize,
}

impl ClipVisualEncoder {
    /// Encode an image to its class-token embedding.
    ///
    /// # Shapes
    /// - input: `[batch, in_channels, height, width]`
    /// - output: `[batch, embed_dim]`
    pub fn forward(&self, image: Tensor<4>) -> Tensor<2> {
        self.forward_with_features(image, true)
            .cls
            .expect("cls requested")
    }

    /// Encode an image and return the patch-token features after each
    /// transformer block. The class token is stripped from each level.
    /// The post-`ln_post` cls embedding is computed only when
    /// `return_cls` is true; A-FINE itself does not use it.
    ///
    /// # Shapes
    /// - input: `[batch, in_channels, height, width]`
    /// - features: `Vec` of length `num_layers`, each
    ///   `[batch, num_patches, embed_dim]`
    /// - cls (when present): `[batch, embed_dim]`
    pub fn forward_with_features(&self, image: Tensor<4>, return_cls: bool) -> ClipOutput {
        let [batch, _, height, width] = image.dims();
        assert_eq!(
            height % self.patch_size,
            0,
            "image height ({}) must be a multiple of patch_size ({})",
            height,
            self.patch_size
        );
        assert_eq!(
            width % self.patch_size,
            0,
            "image width ({}) must be a multiple of patch_size ({})",
            width,
            self.patch_size
        );

        let embed = self.embed_dim;

        // Patch-embed: [B, C, H, W] -> [B, embed, H/p, W/p].
        let x = self.patch_embed.forward(image);
        let [_, _, h_out, w_out] = x.dims();
        let num_patches = h_out * w_out;
        let seq_len = num_patches + 1;

        // Flatten spatial axes, then put the patch axis before the channel
        // axis: [B, embed, N] -> [B, N, embed].
        let x = x.reshape([batch, embed, num_patches]).swap_dims(1, 2);

        // Prepend the CLS token. Reshape [embed] -> [1, 1, embed], then
        // broadcast-expand to [B, 1, embed] before concatenating along the
        // sequence axis.
        let cls = self
            .class_token
            .val()
            .reshape([1, 1, embed])
            .expand([batch, 1, embed]);
        let x = Tensor::cat(vec![cls, x], 1);

        // Add positional embedding. The pos-embed param is initialised to
        // match `seq_len` for the configured image_size; runtime mismatches
        // are caught here. Bicubic resize of a checkpointed pos-embed (e.g.
        // 50 -> 65 when loading a 224-trained checkpoint at 256) is handled
        // in the weights loader, not here.
        let pos_seq = self.positional_embedding.dims()[0];
        assert_eq!(
            pos_seq, seq_len,
            "positional_embedding length {} does not match runtime sequence length {}",
            pos_seq, seq_len
        );
        let pos = self.positional_embedding.val().reshape([1, seq_len, embed]);
        let x = x + pos;

        let mut x = self.ln_pre.forward(x);

        // Run the transformer stack, collecting patch features (CLS
        // dropped) after each block. A-FINE consumes all twelve.
        let mut features = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            x = block.forward(x);
            let patch_features = x.clone().slice([0..batch, 1..seq_len, 0..embed]);
            features.push(patch_features);
        }

        // ln_post is applied only to the class-token output. Skip both
        // the slice and the LayerNorm when the caller doesn't need it.
        let cls = return_cls.then(|| {
            let cls = x.slice([0..batch, 0..1, 0..embed]).reshape([batch, embed]);
            self.ln_post.forward(cls)
        });

        ClipOutput { features, cls }
    }
}

/// Configuration for a single transformer block.
#[derive(Config, Debug)]
pub(crate) struct TransformerBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub mlp_dim: usize,
}

impl TransformerBlockConfig {
    pub(crate) fn init(&self, device: &Device) -> TransformerBlock {
        TransformerBlock {
            ln_1: LayerNormConfig::new(self.d_model).init(device),
            attn: ClipQkvAttentionConfig::new(self.d_model, self.n_heads).init(device),
            ln_2: LayerNormConfig::new(self.d_model).init(device),
            mlp: TransformerMlpConfig::new(self.d_model, self.mlp_dim).init(device),
        }
    }
}

/// Pre-norm transformer block: `x = x + attn(ln_1(x))`, then
/// `x = x + mlp(ln_2(x))`. Matches CLIP's encoder block exactly.
#[derive(Module, Debug)]
pub(crate) struct TransformerBlock {
    pub(crate) ln_1: LayerNorm,
    pub(crate) attn: ClipQkvAttention,
    pub(crate) ln_2: LayerNorm,
    pub(crate) mlp: TransformerMlp,
}

impl TransformerBlock {
    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let attn_out = self.attn.forward(self.ln_1.forward(x.clone()));
        let x = x + attn_out;
        let mlp_out = self.mlp.forward(self.ln_2.forward(x.clone()));
        x + mlp_out
    }
}

/// Configuration for [`TransformerMlp`].
#[derive(Config, Debug)]
pub(crate) struct TransformerMlpConfig {
    pub d_model: usize,
    pub mlp_dim: usize,
}

impl TransformerMlpConfig {
    pub(crate) fn init(&self, device: &Device) -> TransformerMlp {
        TransformerMlp {
            c_fc: LinearConfig::new(self.d_model, self.mlp_dim)
                .with_bias(true)
                .init(device),
            activation: QuickGelu,
            c_proj: LinearConfig::new(self.mlp_dim, self.d_model)
                .with_bias(true)
                .init(device),
        }
    }
}

/// Two-layer MLP with [`QuickGelu`] in between. CLIP's `c_fc` and `c_proj`
/// names are preserved so the checkpoint maps without remapping.
#[derive(Module, Debug)]
pub(crate) struct TransformerMlp {
    pub(crate) c_fc: Linear,
    pub(crate) activation: QuickGelu,
    pub(crate) c_proj: Linear,
}

impl TransformerMlp {
    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.c_fc.forward(x);
        let x = self.activation.forward(x);
        self.c_proj.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn clip_visual_encoder_forward_shape() {
        let device = Default::default();
        let encoder = ClipVisualEncoderConfig::new()
            .with_image_size(64)
            .with_num_layers(2)
            .init(&device);

        let image = Tensor::<4>::random([1, 3, 64, 64], Distribution::Default, &device);
        let cls = encoder.forward(image);

        assert_eq!(cls.dims(), [1, 768]);
    }

    #[test]
    fn clip_visual_encoder_forward_with_features_shape() {
        let device = Default::default();
        let encoder = ClipVisualEncoderConfig::new()
            .with_image_size(64)
            .with_num_layers(3)
            .init(&device);

        let image = Tensor::<4>::random([2, 3, 64, 64], Distribution::Default, &device);
        let ClipOutput { features, cls } = encoder.forward_with_features(image, true);
        let cls = cls.expect("cls requested");

        assert_eq!(cls.dims(), [2, 768]);
        assert_eq!(features.len(), 3);
        // 64x64 image at patch_size=32 -> 2x2 = 4 patches.
        for level in &features {
            assert_eq!(level.dims(), [2, 4, 768]);
        }
    }
}
