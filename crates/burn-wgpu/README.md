# Burn WGPU Backend

[Burn](https://github.com/tracel-ai/burn) WGPU backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-wgpu.svg)](https://crates.io/crates/burn-wgpu)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-wgpu/blob/master/README.md)

This crate provides a WGPU backend for [Burn](https://github.com/tracel-ai/burn) using the
[wgpu](https://github.com/gfx-rs/wgpu).

The backend supports Vulkan, Metal, DirectX11/12, OpenGL, WebGPU.

## Usage Example

```rust
#[cfg(feature = "wgpu")]
mod wgpu {
    use burn_autodiff::Autodiff;
    use burn_wgpu::{Wgpu, WgpuDevice};
    use mnist::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu<f32, i32>>>(device);
    }
}
```

## Configuration

You can set `BURN_WGPU_MAX_TASKS` to a positive integer that determines how many computing tasks are
submitted in batches to the graphics API.

## Alternative SPIR-V backend

When targeting Vulkan, the `spirv` feature flag can be enabled to enable the SPIR-V compiler backend,
which performs significantly better than WGSL. This is especially true for matrix multiplication,
where SPIR-V can make use of TensorCores and run at `f16` precision. This isn't currently supported
by WGSL.
The compiler can also be selected at runtime by setting the corresponding generic parameter to
either `SpirV` or `Wgsl`.

## Platform Support

| Option    | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :-------- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Metal     | No  | Yes |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Vulkan    | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| OpenGL    | No  | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| WebGpu    | No  | Yes |  No   |  No   |   No    |   No    | No  | Yes  |
| Dx11/Dx12 | No  | Yes |  No   |  No   |   Yes   |   No    | No  |  No  |
