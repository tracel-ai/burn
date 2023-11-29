# Burn Torch Backend

[Burn](https://github.com/burn-rs/burn) Torch backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-tch.svg)](https://crates.io/crates/burn-tch)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/burn-rs/burn-tch/blob/master/README.md)

This crate provides a Torch backend for [Burn](https://github.com/burn-rs/burn) utilizing the
[tch-rs](https://github.com/LaurentMazare/tch-rs) crate, which offers a Rust interface to the
[PyTorch](https://pytorch.org/) C++ API.

The backend supports CPU (multithreaded), [CUDA](https://pytorch.org/docs/stable/notes/cuda.html)
(multiple GPUs), and [MPS](https://pytorch.org/docs/stable/notes/mps.html) devices (MacOS).

## Usage Example

```rust
#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn_autodiff::Autodiff;
    use burn_tch::{LibTorch, LibTorchDevice};
    use mnist::training;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        training::run::<Autodiff<LibTorch<f32>>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn_autodiff::Autodiff;
    use burn_tch::{LibTorch, LibTorchDevice};
    use mnist::training;

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        training::run::<Autodiff<LibTorch<f32>>>(device);
    }
}
```

### Platform Support

| Option | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :----- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| CPU    | Yes | No  |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| CUDA   | No  | Yes |  Yes  |  No   |   Yes   |   No    | No  |  No  |
| MPS    | No  | Yes |  No   |  Yes  |   No    |   No    | No  |  No  |
| Vulkan | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | No  |  No  |
