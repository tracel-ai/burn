use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::conv::{Conv3d, Conv3dConfig, ConvTranspose3d, ConvTranspose3dConfig};
use crate::tensor::{Tensor, backend::Backend};

/// Configuration for 3D video patch embedding.
#[derive(Config, Debug)]
pub struct VideoPatchEmbeddingConfig {
    /// Input channels (e.g., RGB = 3).
    pub in_channels: usize,
    /// Output embedding dimension per patch.
    pub embed_dim: usize,
    /// Patch size [t, h, w].
    pub patch_size: [usize; 3],
    /// Stride [t, h, w]; defaults to `patch_size`.
    #[config(default = "None")]
    pub stride: Option<[usize; 3]>,
    /// Include bias in the convolution.
    #[config(default = true)]
    pub bias: bool,
}

/// 3D patch embedding using a Conv3d with kernel=stride=patch.
#[derive(Module, Debug)]
pub struct VideoPatchEmbedding<B: Backend> {
    conv: Conv3d<B>,
    patch: [usize; 3],
    stride: [usize; 3],
}

impl VideoPatchEmbeddingConfig {
    /// Initialize a `VideoPatchEmbedding` module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> VideoPatchEmbedding<B> {
        let stride = self.stride.unwrap_or(self.patch_size);
        let conv = Conv3dConfig::new([self.in_channels, self.embed_dim], self.patch_size)
            .with_stride(stride)
            .with_bias(self.bias)
            .init::<B>(device);
        VideoPatchEmbedding {
            conv,
            patch: self.patch_size,
            stride,
        }
    }
}

impl<B: Backend> VideoPatchEmbedding<B> {
    /// Compute grid sizes (F_p, H_p, W_p) for the given input shape.
    pub fn grid_sizes(&self, depth: usize, height: usize, width: usize) -> [usize; 3] {
        let [st, sh, sw] = self.stride;
        let [kt, kh, kw] = self.patch;
        // Valid conv assumption
        let fp = depth.saturating_sub(kt) / st + 1;
        let hp = height.saturating_sub(kh) / sh + 1;
        let wp = width.saturating_sub(kw) / sw + 1;
        [fp, hp, wp]
    }

    /// Forward to tokens `[B, N, D]` (N = F_p×H_p×W_p, D = embed_dim).
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 3> {
        let [b, _c, _f, _h, _w] = x.dims();
        let y = self.conv.forward(x); // [B, D, Fp, Hp, Wp]
        let [_, d, fp, hp, wp] = y.dims();
        // [B, N, D]
        y.reshape([b, d, fp * hp * wp]).swap_dims(1, 2)
    }

    /// Forward to `[B, D, F_p, H_p, W_p]`.
    pub fn forward_5d(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        self.conv.forward(x)
    }
}

/// Configuration for reconstructing video from patch tokens using ConvTranspose3d.
#[derive(Config, Debug)]
pub struct VideoUnpatchifyConfig {
    /// Output channels (e.g., RGB = 3).
    pub out_channels: usize,
    /// Patch size [t, h, w] used during embedding.
    pub patch_size: [usize; 3],
    /// Stride [t, h, w]; defaults to `patch_size`.
    #[config(default = "None")]
    pub stride: Option<[usize; 3]>,
    /// Include bias in the transposed convolution.
    #[config(default = true)]
    pub bias: bool,
}

/// 3D unpatchify using ConvTranspose3d. Expects tokens reshaped to `[B, D, F_p, H_p, W_p]`.
#[derive(Module, Debug)]
pub struct VideoUnpatchify<B: Backend> {
    deconv: ConvTranspose3d<B>,
    stride: [usize; 3],
}

impl VideoUnpatchifyConfig {
    /// Initialize a `VideoUnpatchify` module.
    pub fn init<B: Backend>(&self, embed_dim: usize, device: &B::Device) -> VideoUnpatchify<B> {
        let stride = self.stride.unwrap_or(self.patch_size);
        let deconv = ConvTranspose3dConfig::new([embed_dim, self.out_channels], self.patch_size)
            .with_stride(stride)
            .with_bias(self.bias)
            .init::<B>(device);
        VideoUnpatchify { deconv, stride }
    }
}

impl<B: Backend> VideoUnpatchify<B> {
    /// Unpatchify from tokens `[B, N, D]` given grid sizes `[F_p, H_p, W_p]`.
    pub fn forward(&self, tokens: Tensor<B, 3>, grid_sizes: [usize; 3]) -> Tensor<B, 5> {
        let [b, n, d] = tokens.dims();
        let [fp, hp, wp] = grid_sizes;
        assert_eq!(n, fp * hp * wp, "tokens length must match grid sizes");
        let x = tokens.swap_dims(1, 2).reshape([b, d, fp, hp, wp]);
        self.deconv.forward(x)
    }

    /// Unpatchify from `[B, D, F_p, H_p, W_p]`.
    pub fn forward_5d(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
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
        let in_ch = 2;
        let out_ch = 2;
        let embed = 8;
        let patch = [2, 2, 2];
        let x =
            Tensor::<TestBackend, 5>::random([1, in_ch, 4, 4, 4], Distribution::Default, &device);

        let pe = VideoPatchEmbeddingConfig::new(in_ch, embed, patch).init::<TestBackend>(&device);
        let tokens = pe.forward(x);
        let [b, n, d] = tokens.dims();
        assert_eq!(b, 1);
        assert_eq!(d, embed);

        let gs = pe.grid_sizes(4, 4, 4);
        assert_eq!(n, gs[0] * gs[1] * gs[2]);

        let up = VideoUnpatchifyConfig::new(out_ch, patch)
            .with_stride(None)
            .init::<TestBackend>(embed, &device);
        let y = up.forward(tokens, gs);
        let [bb, cc, ff, hh, ww] = y.dims();
        assert_eq!([bb, cc, ff, hh, ww], [1, out_ch, 4, 4, 4]);
    }
}
