# Burn Autodiff

> Reverse-mode automatic differentiation for [Burn](https://github.com/tracel-ai/burn), as a backend decorator

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-autodiff.svg)](https://crates.io/crates/burn-autodiff)
[![Documentation](https://docs.rs/burn-autodiff/badge.svg)](https://docs.rs/burn-autodiff)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

`Autodiff<B>` wraps any backend `B` and records the operations needed to compute gradients. Only
first-order derivatives are supported.

## Usage

Most applications never name the `Autodiff` type. Enable Burn's `autodiff` feature (also enabled
by `train`) and turn autodiff on for a device before creating parameters and inputs:

```toml
burn = { version = "0.22", features = ["autodiff", "wgpu"] }
```

```rust,ignore
use burn::tensor::{Device, Tensor};

let device = Device::wgpu(Default::default()).autodiff();
let x = Tensor::<2>::ones([2, 2], &device).require_grad();

let grads = (x.clone() * 3.0).sum().backward();
let x_grad = x.grad(&grads).unwrap();
```

Use this crate directly when writing a backend extension that needs a custom backward pass.

## Gradient Checkpointing

`device.autodiff().gradient_checkpointing()` recomputes cheap operations during the backward
pass instead of storing their outputs, trading compute for memory. At the type level, this is the
`BalancedCheckpointing` strategy; the default, `NoCheckpointing`, stores every activation the
backward pass needs.

## Feature Flags

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- `tracing`: instrument operations with the `tracing` crate.

See the [autodiff chapter](https://burn.dev/books/burn/building-blocks/autodiff.html) of the Burn
Book for how autodiff works at the tensor level.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
