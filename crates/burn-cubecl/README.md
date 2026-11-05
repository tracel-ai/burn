# Burn CubeCL Backend

> The [Burn](https://github.com/tracel-ai/burn) backend for every [CubeCL](https://github.com/tracel-ai/cubecl) runtime

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-cubecl.svg)](https://crates.io/crates/burn-cubecl)
[![Documentation](https://docs.rs/burn-cubecl/badge.svg)](https://docs.rs/burn-cubecl)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

`CubeBackend` implements Burn's tensor operations as CubeCL kernels, compiled just in time for the
device they run on. CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu and the CubeCL CPU runtime all share
this one backend type; a tensor's device says which runtime it uses.

## Usage

Applications select a runtime with a Burn feature and a `Device` constructor:

| Runtime | Burn feature | Device                               | Crate                     |
| ------- | ------------ | ------------------------------------ | ------------------------- |
| CUDA    | `cuda`       | `Device::cuda(0)`                    | [burn-cuda](https://github.com/tracel-ai/burn/tree/main/crates/burn-cuda) |
| ROCm    | `rocm`       | `Device::rocm(0)`                    | [burn-rocm](https://github.com/tracel-ai/burn/tree/main/crates/burn-rocm) |
| wgpu    | `wgpu`       | `Device::wgpu(Default::default())`   | [burn-wgpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) |
| Metal   | `metal`      | `Device::metal(Default::default())`  | [burn-wgpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) |
| Vulkan  | `vulkan`     | `Device::vulkan(Default::default())` | [burn-wgpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) |
| WebGPU  | `webgpu`     | `Device::webgpu(Default::default())` | [burn-wgpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu) |
| CPU     | `cpu`        | `Device::cpu()`                      | [burn-cpu](https://github.com/tracel-ai/burn/tree/main/crates/burn-cpu)   |

Use this crate directly to write custom kernels: `kernel` and `ops` hold the building blocks, and
`cubecl` is re-exported so kernels use the same CubeCL version as the backend. See the
[custom CubeCL kernel](https://burn.dev/books/burn/advanced/backend-extension/custom-cubecl-kernel.html)
chapter of the Burn Book.

## Feature Flags

- `cuda`, `hip`, `wgpu`, `metal`, `vulkan`, `webgpu`, `cpu`: compile in a CubeCL runtime.
- `fusion` (default): kernel fusion through [burn-fusion](https://github.com/tracel-ai/burn/tree/main/crates/burn-fusion).
- `autotune` (default): benchmark kernel variants at runtime and keep the fastest.
- `fft`: FFT kernels.
- `template`: launch hand-written, non-JIT kernels.
- `tracing`: instrument operations with the `tracing` crate.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
