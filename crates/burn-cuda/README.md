# Burn CUDA Backend

[Burn](https://github.com/tracel-ai/burn) CUDA backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-cuda.svg)](https://crates.io/crates/burn-cuda)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-cuda/blob/master/README.md)

This crate provides a CUDA backend for [Burn](https://github.com/tracel-ai/burn) using the
[cubecl](https://github.com/tracel-ai/cubecl.git) and [cudarc](https://github.com/coreylowman/cudarc.git)
crates.

## Usage Example

```rust
#[cfg(feature = "cuda")]
mod cuda {
    use burn_autodiff::Autodiff;
    use burn_cuda::{Cuda, CudaDevice};
    use mnist::training;

    pub fn run() {
        let device = CudaDevice::default();
        training::run::<Autodiff<Cuda<f32, i32>>>(device);
    }
}
```

## Dependencies

Requires CUDA 12.x to be installed and on the `PATH`.