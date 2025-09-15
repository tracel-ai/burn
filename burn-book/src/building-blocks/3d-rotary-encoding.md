# 3D Rotary Positional Encoding (F/H/W)

Rotary encoding over 3D grids applies per-axis rotations for frames (F), height (H), and width (W). It’s useful for video transformers and diffusion models that operate on patchified (F×H×W) tokens.

## API

- `Rope3dEncodingConfig::new(max_f, max_h, max_w, d_head)`
  - Optional: `.with_half_dim_split([f_pairs, h_pairs, w_pairs])` where `f_pairs + h_pairs + w_pairs = d_head / 2`.
- `Rope3dEncoding::apply(x, [F, H, W], start_frame)`
  - `x`: `[batch, seq, n_heads, d_head]` with `seq = F×H×W`, `d_head` even.
  - `start_frame`: offsets the F-axis positions for streaming.

## Example

```rust
use burn_core::nn::{Rope3dEncodingConfig};
use burn_core::tensor::{Distribution, Tensor};

let device = Default::default();
let d_head = 64; // per-head dim, even
let rope3d = Rope3dEncodingConfig::new(512, 32, 32, d_head).init::<B>(&device);

let f = 8usize; let h = 8usize; let w = 8usize; // seq = f*h*w
let x = Tensor::<B, 4>::random([1, f*h*w, 12, d_head], Distribution::Default, &device);
let y = rope3d.apply(x, [f, h, w], /*start_frame*/ 0);
```

## Notes

- Channel split: The head’s half-dimension is divided across F/H/W. By default, it’s near-equal (favoring F if not divisible by 3). You can pass an explicit split to match model conventions.
- Streaming: `start_frame` allows chunked/streaming inference to advance the frame offset without rebuilding frequencies.
- Performance: The implementation is correctness-first using tensor ops; backends can optimize gathers and broadcasting as needed.

## References

- [Rotary Embeddings (RoFormer, arXiv)](https://arxiv.org/abs/2104.09864)
- 3D RoPE usage in recent video models, e.g., [CausVid (arXiv)](https://arxiv.org/abs/2412.07772)
