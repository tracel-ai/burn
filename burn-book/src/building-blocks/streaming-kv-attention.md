# Streaming KV Multi-Head Attention

This page introduces a streaming multi-head attention (MHA) module with a rolling K/V cache and an optional local attention window with sink tokens. It enables long-sequence, blockwise causal inference and serves as a stable API target for backend optimizations.

## Why

- Efficient long-context inference for video/audio/world models and LLMs.
- Backend-agnostic API surface in `burn-core`, allowing GPU backends (CubeCL/WGPU/CUDA) to implement optimized sliding-window kernels later without user-facing changes.

## Core Types

- `StreamingMultiHeadAttention<B>`
  - Q/K/V projections and output projection, similar to `MultiHeadAttention`.
  - Adds `forward_streaming` that consumes chunks and updates a rolling K/V cache.
- `StreamingMhaCache<B>`
  - Stores K/V as `[batch, cache_len, n_heads, head_dim]`.
  - Tracks indices: `global_end_index`, `local_end_index` and config: `sink_tokens`, `cache_len`.
- `StreamingParams<'a, B>`
  - Parameters for a streaming call: `rope: Option<&RotaryEncoding<B>>`, `start_pos`, and `window_len: Option<usize>`.

## Forward Flow

1. Project Q/K/V as in standard MHA. Optionally apply RoPE with `start_pos` to Q and K.
2. Update the rolling cache with the new K/V tokens, evicting older tokens beyond capacity while preserving `sink_tokens`.
3. Select active keys: full-causal (no window) or `sink_tokens + window_len`.
4. Compute attention over the selected keys and apply the output projection.

## Mask Utility

- `generate_windowed_causal_mask(batch, seq_len, window_len, sink_tokens, &device)` builds a dense boolean mask that matches the windowed causal behavior (useful for training or baselines). Semantics: `true` means “masked”.

## RoPE Usage

- Use `RotaryEncoding` initialized with `d_model = head_dim` (per-head). Apply with `start_pos` matching the absolute token index of the first token in the current chunk.

```rust
use burn_core::nn::attention::{
    AttnWindow, StreamingMhaCache, StreamingMultiHeadAttentionConfig, StreamingParams,
};
use burn_core::nn::rope_encoding::RotaryEncodingConfig;

let device = Default::default();
let d_model = 1536;
let n_heads = 12;
let head_dim = d_model / n_heads;

let attn = StreamingMultiHeadAttentionConfig::new(d_model, n_heads)
    .with_dropout(0.0)
    .init::<B>(&device);

let mut cache = StreamingMhaCache::new(&device, batch, /*cache_len*/ 32768, n_heads, head_dim, /*sink*/ 0);
let rope = RotaryEncodingConfig::new(/*max_seq*/ 65536, head_dim).init::<B>(&device);

let params = StreamingParams { rope: Some(&rope), start_pos: 0, window: AttnWindow::Window(4096) };
let y = attn.forward_streaming(x_chunk, &mut cache, params);
```

## Notes

- The reference implementation is correctness-first using tensor ops. Backends can add fused kernels later without changing the API.
- For per-head RoPE, ensure `RotaryEncoding` uses `d_model = head_dim`.

## Where to Find It

- Code: `crates/burn-core/src/nn/attention/streaming.rs`
- Mask: `crates/burn-core/src/nn/attention/mask.rs` (`generate_windowed_causal_mask`)
- Integration tests: `crates/burn-core/tests/attention_streaming.rs`
