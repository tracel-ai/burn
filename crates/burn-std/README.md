# Burn Standard Library

> Core types and utilities shared across the [Burn](https://github.com/tracel-ai/burn) crates

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-std.svg)](https://crates.io/crates/burn-std)
[![Documentation](https://docs.rs/burn-std/badge.svg)](https://docs.rs/burn-std)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

- `Shape`, slicing (`s!`) and indexing helpers.
- `DType`, the element traits and conversions, and `TensorData`, the backend-independent
  representation of tensor contents.
- `Distribution` for random tensor initialization.
- `DeviceSettings`, per-device defaults such as dtypes.
- Quantization schemes, `Bytes`, identifiers and errors.
- `config`: runtime configuration read from `burn.toml`.
- `network`: file downloads with a progress bar (`network` feature).

Applications use these through `burn::tensor`. This crate supports both `std` and `no_std`
environments and must compile with `cargo build --no-default-features` as well.

## Feature Flags

- `std` (default): standard library support.
- `network`: file downloads.
- `tracing`: instrument operations with the `tracing` crate.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
