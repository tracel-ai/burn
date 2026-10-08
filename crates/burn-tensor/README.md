# Burn Tensor

> [Burn](https://github.com/tracel-ai/burn)'s tensor API

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-tensor.svg)](https://crates.io/crates/burn-tensor)
[![Documentation](https://docs.rs/burn-tensor/badge.svg)](https://docs.rs/burn-tensor)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

`Tensor<D, K>` is a tensor of rank `D` and kind `K` (`Float` by default, `Int` or `Bool`). Its
`Device` decides which backend runs it, at runtime: the same `Tensor<2>` can live on CUDA, wgpu or
the CPU, and code that uses tensors has no backend type parameter. Applications use this crate
through `burn::tensor`.

```rust,ignore
use burn::tensor::{Device, Tensor, s};

let device = Device::default();
let x = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
let y = x.clone().matmul(x.transpose()).exp();
let first_row = y.slice(s![0..1, ..]);
```

- `Device`: backend selection, configuration and enumeration. A backend's constructor
  (`Device::cuda`, `Device::wgpu`, `Device::flex`, ...) exists when its feature is enabled.
- `TensorData`, `DType` and `Shape`: tensor contents and metadata, independent of any backend.
- `activation`, `loss` and `module`: functional forms of activations, losses and neural network
  operations such as convolution and pooling.
- `quantization`, `grid` and `distributed`: quantized tensors, grid sampling and collective
  operations across devices.
- `einsum!`, `assert_shape!` and `s!`: macros for Einstein summation, shape checks and slicing.

With the `autodiff` feature, create tensors on `device.autodiff()` and call `require_grad()` on
source leaves whose gradients you need.

See the [tensor chapter](https://burn.dev/books/burn/building-blocks/tensor.html) of the Burn Book.

## Feature Flags

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- Backends: `cuda`, `rocm`, `wgpu`, `metal`, `vulkan`, `webgpu`, `cpu`, `flex`, and the deprecated
  `tch`.
- `autodiff`, `fusion`: backend decorators.
- `remote`, `remote-server`, `remote-websocket`: devices hosted by another machine.
- `capture`: record operation graphs instead of executing them.
- `extension`: access to backend primitives for backend extensions.
- `tracing`: instrument operations with the `tracing` crate.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
