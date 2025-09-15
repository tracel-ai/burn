# Attention Masks (1D Helpers)

This page provides small utilities for common 1D sequence masking patterns.

## Why

- Many text/audio models need simple helpers to create padding masks from sequence lengths or to generate causal/chunked masks for streaming encoders.
- Keeping these utilities in `burn-core` avoids ad‑hoc masking logic scattered across projects.

## API

- `lengths_to_mask(lengths: &[usize], max_len: usize, &device) -> BoolTensor [B, L]`
  - True marks padding positions (masked), false marks valid tokens.

- `generate_causal_mask_1d(seq_len, &device) -> BoolTensor [L, L]`
  - Lower‑triangular mask (true = masked) suitable for causal attention without batching.

- `generate_chunked_causal_mask_1d(seq_len, chunk_size, num_left_chunks, &device) -> BoolTensor [L, L]`
  - Conformer‑style streaming mask: each row `i` can attend within a local window of chunks to the left and its own chunk; future positions remain masked.

## Notes

- Mask semantics: consistent with other Burn masks — `true` means masked.
- For batched causal masks, prefer `generate_autoregressive_mask(batch, length, &device)`.

