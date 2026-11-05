# Burn Derive

> Derive macros for [Burn](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-derive.svg)](https://crates.io/crates/burn-derive)
[![Documentation](https://docs.rs/burn-derive/badge.svg)](https://docs.rs/burn-derive)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Use these through `burn`, which re-exports them; this crate is not meant to be a direct dependency.

- `#[derive(Module)]` implements `Module` for a struct or enum of modules, parameters and constants.
  `#[module(skip)]` excludes a field.
- `#[derive(Config)]` makes a struct a serializable configuration with a generated `new`
  constructor and `with_*` setters for optional fields and fields marked `#[config(default = ...)]`.
- `#[derive(RecordState)]` decomposes an optimizer or scheduler state into named tensors and scalars
  for the burnpack format.

The crate also implements the `assert_shape!`, `debug_assert_shape!` and `einsum!` tensor macros.

```rust,ignore
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct MlpConfig {
    hidden: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct Mlp {
    linear: nn::Linear,
    dropout: nn::Dropout,
}
```

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
