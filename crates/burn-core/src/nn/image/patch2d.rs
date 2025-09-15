use crate as burn;

use crate::module::Module;
use crate::config::Config;
use crate::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use crate::tensor::{Tensor, backend::Backend};
//

/// Configuration for 2D image patch embedding.
#[derive(Config, Debug)]
pub struct ImagePatchEmbeddingConfig {
    /// Input channels (e.g., RGB = 3).
    pub in_channels: usize,
    /// Output embedding dimension per patch.
    pub embed_dim: usize,
    /// Patch size [h, w].
    pub patch_size: [usize; 2],
    /// Stride [h, w]; defaults to `patch_size`.
    #[config(default = "None")]
    pub stride: Option<[usize; 2]>,
    /// Include bias in the convolution.
    #[config(default = true)]
    pub bias: bool,
}

/// 2D patch embedding using Conv2d with kernel=stride=patch.
#[derive(Module, Debug)]
pub struct ImagePatchEmbedding<B: Backend> {
    conv: Conv2d<B>,
    patch: [usize; 2],
    stride: [usize; 2],
}

impl ImagePatchEmbeddingConfig {
    /// Initialize the module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ImagePatchEmbedding<B> {
        let stride = self.stride.unwrap_or(self.patch_size);
        let conv = Conv2dConfig::new([self.in_channels, self.embed_dim], self.patch_size)
            .with_stride(stride)
            .with_bias(self.bias)
            .init::<B>(device);
        ImagePatchEmbedding { conv, patch: self.patch_size, stride }
    }
}

impl<B: Backend> ImagePatchEmbedding<B> {
    /// Compute grid sizes (H_p, W_p) for the given input shape.
    pub fn grid_sizes(&self, height: usize, width: usize) -> [usize; 2] {
        let [sh, sw] = self.stride;
        let [kh, kw] = self.patch;
        let hp = height.saturating_sub(kh) / sh + 1;
        let wp = width.saturating_sub(kw) / sw + 1;
        [hp, wp]
    }

    /// Forward to tokens `[B, N, D]` (N = H_p×W_p, D = embed_dim).
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, _c, _h, _w] = x.dims();
        let y = self.conv.forward(x); // [B, D, Hp, Wp]
        let [_, d, hp, wp] = y.dims();
        y.reshape([b, d, hp * wp]).swap_dims(1, 2)
    }

    /// Forward to `[B, D, H_p, W_p]`.
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(x)
    }
}

/// Configuration for reconstructing images from patch tokens using ConvTranspose2d.
#[derive(Config, Debug)]
pub struct ImageUnpatchifyConfig {
    /// Output channels (e.g., RGB = 3).
    pub out_channels: usize,
    /// Patch size [h, w] used during embedding.
    pub patch_size: [usize; 2],
    /// Stride [h, w]; defaults to `patch_size`.
    #[config(default = "None")]
    pub stride: Option<[usize; 2]>,
    /// Include bias in the transposed convolution.
    #[config(default = true)]
    pub bias: bool,
}

/// 2D unpatchify using ConvTranspose2d. Expects tokens reshaped to `[B, D, H_p, W_p]`.
#[derive(Module, Debug)]
pub struct ImageUnpatchify<B: Backend> {
    deconv: ConvTranspose2d<B>,
    stride: [usize; 2],
}

impl ImageUnpatchifyConfig {
    /// Initialize the module.
    pub fn init<B: Backend>(&self, embed_dim: usize, device: &B::Device) -> ImageUnpatchify<B> {
        let stride = self.stride.unwrap_or(self.patch_size);
        let deconv =
            ConvTranspose2dConfig::new([embed_dim, self.out_channels], self.patch_size)
                .with_stride(stride)
                .with_bias(self.bias)
                .init::<B>(device);
        ImageUnpatchify { deconv, stride }
    }
}

impl<B: Backend> ImageUnpatchify<B> {
    /// Unpatchify from tokens `[B, N, D]` given grid sizes `[H_p, W_p]`.
    pub fn forward(&self, tokens: Tensor<B, 3>, grid_sizes: [usize; 2]) -> Tensor<B, 4> {
        let [b, n, d] = tokens.dims();
        let [hp, wp] = grid_sizes;
        assert_eq!(n, hp * wp, "tokens length must match grid sizes");
        let x = tokens.swap_dims(1, 2).reshape([b, d, hp, wp]);
        self.deconv.forward(x)
    }

    /// Unpatchify from `[B, D, H_p, W_p]`.
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.deconv.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::tensor::{Distribution, Tensor};

    #[test]
    fn shapes_roundtrip() {
        let device = Default::default();
        let in_ch = 3;
        let out_ch = 3;
        let embed = 8;
        let patch = [4, 4];
        let x = Tensor::<TestBackend, 4>::random([1, in_ch, 16, 16], Distribution::Default, &device);

        let pe = ImagePatchEmbeddingConfig {
            in_channels: in_ch,
            embed_dim: embed,
            patch_size: patch,
            stride: None,
            bias: true,
        }
        .init::<TestBackend>(&device);
        let tokens = pe.forward(x);
        let [b, n, d] = tokens.dims();
        assert_eq!(b, 1);
        assert_eq!(d, embed);

        let gs = pe.grid_sizes(16, 16);
        assert_eq!(n, gs[0] * gs[1]);

        let up = ImageUnpatchifyConfig {
            out_channels: out_ch,
            patch_size: patch,
            stride: None,
            bias: true,
        }
            .init::<TestBackend>(embed, &device);
        let y = up.forward(tokens, gs);
        let [bb, cc, hh, ww] = y.dims();
        assert_eq!([bb, cc, hh, ww], [1, out_ch, 16, 16]);
    }
}
