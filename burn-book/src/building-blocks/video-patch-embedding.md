# 3D Video Patch Embedding

Patchify videos into spatio‑temporal tokens and reconstruct with transposed convolution. Useful for video transformers and diffusion models.

## API

- `VideoPatchEmbeddingConfig::new(in_channels, embed_dim, patch_size)`
  - Optional: `.with_stride(Some([t,h,w]))` (defaults to `patch_size`), `.with_bias(true/false)`.
  - `init::<B>(&device)` returns `VideoPatchEmbedding<B>`.
  - `forward(x)` expects `[B, C_in, F, H, W]`, returns `[B, N, D]` (N = `F_p×H_p×W_p`).
  - `forward_5d(x)` returns `[B, D, F_p, H_p, W_p]`.
  - `grid_sizes(F, H, W)` computes `[F_p, H_p, W_p]`.

- `VideoUnpatchifyConfig::new(out_channels, patch_size)`
  - Optional: `.with_stride(Some([t,h,w]))` (defaults to `patch_size`), `.with_bias(true/false)`.
  - `init::<B>(embed_dim, &device)` returns `VideoUnpatchify<B>`.
  - `forward(tokens, [F_p, H_p, W_p])` expects `[B, N, D]` and reconstructs `[B, C_out, F, H, W]`.
  - `forward_5d(x)` expects `[B, D, F_p, H_p, W_p]`.

## Example

```rust
use burn_core::nn::{
    VideoPatchEmbeddingConfig, VideoUnpatchifyConfig
};
use burn_core::tensor::{Distribution, Tensor};

let device = Default::default();
let patch = [2, 2, 2];
let pe = VideoPatchEmbeddingConfig::new(3, 96, patch).init::<B>(&device);

let x = Tensor::<B, 5>::random([1, 3, 16, 224, 224], Distribution::Default, &device);
let tokens = pe.forward(x.clone()); // [1, N, 96]
let [fp, hp, wp] = pe.grid_sizes(16, 224, 224);

let up = VideoUnpatchifyConfig::new(3, patch).init::<B>(96, &device);
let y = up.forward(tokens, [fp, hp, wp]); // [1, 3, 16, 224, 224]
```

## Notes

- Uses Conv3d for patchify (kernel=stride=patch) and ConvTranspose3d for unpatchify.
- Defaults assume valid padding. For custom padding/stride, ensure shapes divide as intended.
- Pair with 3D RoPE ([docs](./3d-rotary-encoding.md)) and streaming KV attention for long‑sequence video models.

## References

- Video transformers and patchifying: [TimeSformer (arXiv)](https://arxiv.org/abs/2102.05095), [ViViT (arXiv)](https://arxiv.org/abs/2103.15691), [VideoMAE (arXiv)](https://arxiv.org/abs/2203.12602)
