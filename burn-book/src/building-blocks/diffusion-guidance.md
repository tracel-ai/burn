# Diffusion Guidance Utilities

Small, generic guidance functions for diffusion/flow‑matching pipelines.

## Why

- Classifier‑free guidance (CFG) and related variants are used across many pipelines to trade off fidelity and adherence to conditioning signals.
- Providing lightweight, backend‑agnostic helpers in `burn-core` avoids duplicating the same math in every project.

## API

- `cfg(cond, uncond, strength)` → basic classifier‑free guidance.
- `cfg_double(cond, uncond, only_text, g_text, g_lyric)` → double‑condition guidance.
- `apg(pred_cond, pred_uncond, guidance_scale, buffer, norm_eps)` → momentum‑aided projection with optional norm stabilization.
- `cfg_zero_star(cond, uncond, strength)` → zero‑star variant.

## References

- CFG is widely used in diffusion pipelines (e.g., Stable Diffusion variants).
- Momentum‑aided variants appear in practical codebases; this module provides a minimal, composable version.

