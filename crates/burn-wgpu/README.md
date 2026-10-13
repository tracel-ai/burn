# Burn WGPU Backend

[Burn](https://github.com/tracel-ai/burn) WGPU backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-wgpu.svg)](https://crates.io/crates/burn-wgpu)
[![Documentation](https://docs.rs/burn-wgpu/badge.svg)](https://docs.rs/burn-wgpu)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

This crate provides a WGPU backend for [Burn](https://github.com/tracel-ai/burn) using the
[wgpu](https://github.com/gfx-rs/wgpu).

The backend supports Vulkan, Metal, DirectX 12, OpenGL, and WebGPU.

## Usage Example

For application code, enable Burn's `wgpu` feature and select the device at runtime:

```toml
burn = { version = "0.22", features = ["wgpu"] }
```

```rust
use burn::tensor::{Device, Tensor};

let device = Device::wgpu(Default::default());
let input = Tensor::<2>::ones([2, 3], &device);
let output = input + 1.0;
```

For training, enable `autodiff` (also enabled by `train`) and use `device.autodiff()` before
initializing model parameters and inputs. Tensor and model types have no backend parameter.

## Configuration

Use `Device::wgpu_options()` to configure the runtime before creating tensors:

```rust,ignore
use burn::tensor::{Device, wgpu::WgpuBackend};

let device = Device::wgpu_options()
    .graphics_api(WgpuBackend::Vulkan)
    .tasks_max(32)
    .init()?;
```

Use `.init_async().await?` in browsers or async applications. Options also include
`.memory_config(...)` and `.setup(...)` for existing wgpu handles.
Use `Device::configure` for dtype defaults.

## Graphics API and shader compiler

Enable `vulkan`, `metal`, or `webgpu` and select the corresponding `Device` constructor to target
that graphics API. `AutoCompiler` selects the shader compiler at runtime. There is no `spirv`
feature or compiler type parameter on `Wgpu` in 0.22. Multiple backend features can be enabled
together. The `Vulkan`, `Metal`, and `WebGpu` aliases are deprecated: use `Wgpu` with an explicit
device constructor to select the graphics API.

## Platform Support

| Option | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :----- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Metal  | No  | Yes |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Vulkan | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| OpenGL | No  | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| WebGpu | No  | Yes |  No   |  No   |   No    |   No    | No  | Yes  |
| Dx12   | No  | Yes |  No   |  No   |   Yes   |   No    | No  |  No  |

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
