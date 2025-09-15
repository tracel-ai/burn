# Linear Attention

Linear attention replaces the quadratic softmax attention with a kernel feature map for queries and keys, enabling attention computation in time that scales linearly with sequence length.

## Why

- Long‑sequence inference and streaming workloads benefit from reduced memory/runtime complexity versus standard softmax attention.
- Provides a drop‑in alternative to multi‑head attention for self/cross attention when masks are simple (e.g., padding masks), while staying backend‑agnostic.

## API

- `LinearAttentionConfig::new(d_model, n_heads)` → `init::<B>(&device)`
- `LinearAttention::forward(LinearAttnInput { query, key, value, mask_pad: Option<_> })`
  - Shapes: `query: [B, Sq, D]`, `key/value: [B, Sk, D]`, output: `[B, Sq, D]`
  - Current implementation supports padding masks on keys/values; arbitrary attention masks are not yet supported.

## Example

```rust
use burn::prelude::*;
use burn::nn::attention::{LinearAttentionConfig, LinearAttnInput};

let device = Default::default();
let la = LinearAttentionConfig::new(256, 8).init::<burn_ndarray::NdArray<f32>>(&device);

let x = Tensor::random([1, 128, 256], Distribution::Default, &device);
let y = la.forward(LinearAttnInput::self_attn(x));
```

## References

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (arXiv)](https://arxiv.org/abs/2006.16236)
- [Performer: Rethinking Attention with Performers (arXiv)](https://arxiv.org/abs/2009.14794)

