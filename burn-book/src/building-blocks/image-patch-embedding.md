# 2D Image Patch Embedding

Patchify images into tokens with a Conv2d and reconstruct with ConvTranspose2d. Useful for latent 2D transformers and diffusion models.

## API

- `ImagePatchEmbeddingConfig::new(in_channels, embed_dim, [h, w])`
  - Optional: `.with_stride(Some([h, w]))` (defaults to patch size), `.with_bias(true/false)`.
  - `forward(x: [B, C, H, W]) -> [B, N, D]` where `N = H_p×W_p`.
  - `forward_4d(x) -> [B, D, H_p, W_p]` and `grid_sizes(H, W) -> [H_p, W_p]`.

- `ImageUnpatchifyConfig::new(out_channels, [h, w])`
  - Optional: `.with_stride(Some([h, w]))`.
  - `init(embed_dim, &device)` returns `ImageUnpatchify`.
  - `forward(tokens: [B, N, D], [H_p, W_p]) -> [B, C, H, W]`.

## Example

```rust
use burn::prelude::*;
use burn::nn::image::{ImagePatchEmbeddingConfig, ImageUnpatchifyConfig};

let device = Default::default();
let patch = [4, 4];
let pe = ImagePatchEmbeddingConfig::new(3, 64, patch).init::<burn_ndarray::NdArray<f32>>(&device);

let x = Tensor::random([1, 3, 128, 128], Distribution::Default, &device);
let tokens = pe.forward(x.clone());
let [hp, wp] = pe.grid_sizes(128, 128);

let up = ImageUnpatchifyConfig::new(3, patch).init::<burn_ndarray::NdArray<f32>>(64, &device);
let y = up.forward(tokens, [hp, wp]);
```

## Notes

- Defaults assume valid padding. For custom stride/padding, ensure shapes divide as intended.
- Pairs naturally with positional encodings and attention layers for 2D latents.

