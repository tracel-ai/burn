# Burn CUDA Backend

[Burn](https://github.com/tracel-ai/burn) CUDA backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-cuda.svg)](https://crates.io/crates/burn-cuda)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-cuda/blob/master/README.md)

This crate provides a CUDA backend for [Burn](https://github.com/tracel-ai/burn) using the
[cubecl](https://github.com/tracel-ai/cubecl.git) and [cudarc](https://github.com/coreylowman/cudarc.git)
crates.

## Usage Example

For application code, enable Burn's `cuda` feature and select the device at runtime:

```toml
burn = { version = "0.22", features = ["cuda"] }
```

```rust
use burn::tensor::{Device, Tensor};

let device = Device::cuda(0);
let input = Tensor::<2>::ones([2, 3], &device);
let output = input + 1.0;
```

For training, enable `autodiff` (also enabled by `train`) and use `device.autodiff()` before
initializing model parameters and inputs. Tensor and model types have no backend parameter.
Use `Device::configure` for dtype defaults; the low-level `Cuda` alias no longer takes element
type parameters.

## Dependencies

Requires CUDA 12.x to be installed and on the `PATH`.
