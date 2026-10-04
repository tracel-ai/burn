# Burn CubeCL Fusion

> Fused [CubeCL](https://github.com/tracel-ai/cubecl) kernels for [Burn](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-cubecl-fusion.svg)](https://crates.io/crates/burn-cubecl-fusion)
[![Documentation](https://docs.rs/burn-cubecl-fusion/badge.svg)](https://docs.rs/burn-cubecl-fusion)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

This crate is the CubeCL side of kernel fusion. [burn-fusion](https://github.com/tracel-ai/burn/tree/main/crates/burn-fusion) decides which queued
operations can be grouped; this crate generates and launches one kernel for each group on any
CubeCL runtime.

- `engine` traces a group of operations and compiles it into a single kernel.
- `optim` holds the fused optimizations: element-wise chains, and matmuls and reductions together
  with the element-wise work around them.

[burn-cubecl](https://github.com/tracel-ai/burn/tree/main/crates/burn-cubecl) uses it when its `fusion` feature is on, which is the default.
Applications do not use this crate directly.

## Feature Flags

- `std` (default): standard library support.
- `autotune` (default): benchmark fused kernel variants at runtime and keep the fastest.
- `autotune-checks`: check autotuned variants against each other for correctness.
- `tracing`: instrument operations with the `tracing` crate.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
