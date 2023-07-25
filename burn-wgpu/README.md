# Burn WGPU Backend

[Burn](https://github.com/burn-rs/burn) WGPU backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-wgpu.svg)](https://crates.io/crates/burn-wgpu)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/burn-rs/burn-wgpu/blob/master/README.md)

This crate provides a WGPU backend for [Burn](https://github.com/burn-rs/burn) utilizing the
[wgpu](https://github.com/gfx-rs/wgpu). 

The backend supports Vulkan, Metal, DirectX11/12, OpenGL, WebGPU.

## Usage Example

```rust
#[cfg(feature = "wgpu")]
mod wgpu {
    use burn_autodiff::ADBackendDecorator;
    use burn_wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};
    use mnist::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<ADBackendDecorator<WgpuBackend<AutoGraphicsApi, f32, i32>>>(device);
    }
}
```
