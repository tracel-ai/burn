# Burn CPU Backend

[Burn](https://github.com/tracel-ai/burn) CubeCL CPU backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-cpu.svg)](https://crates.io/crates/burn-cpu)

This crate provides an MLIR based CPU backend for [Burn](https://github.com/tracel-ai/burn) using
the [cubecl](https://github.com/tracel-ai/cubecl.git) crates. It is the CubeCL runtime targeting the
CPU: the same CubeCL kernels that power the CUDA, ROCm, Metal, Vulkan and WebGPU backends are
JIT-compiled to native CPU code through MLIR/LLVM, with kernel fusion and autotuning enabled by
default.

## burn-cpu vs burn-flex

Burn has two independent CPU backends. Neither one is built on top of the other.

|                    | `burn-cpu`                                   | [`burn-flex`](../burn-flex)                   |
| ------------------ | -------------------------------------------- | --------------------------------------------- |
| Implementation     | CubeCL kernels compiled through MLIR/LLVM    | Hand-written Rust kernels, `gemm`, SIMD       |
| Execution          | JIT-compiled, with fusion and autotune       | Eager                                         |
| Native deps        | LLVM/MLIR toolchain                          | None (pure Rust)                              |
| `no_std` / Wasm    | No                                           | Yes                                           |
| `burn` feature     | `cpu`                                        | `flex`                                        |
| Device constructor | `Device::cpu()`                              | `Device::flex()`                              |

Use `burn-cpu` when you want the CubeCL stack (fusion, custom CubeCL kernels shared with the GPU
backends) on the CPU. Use `burn-flex` for a lightweight, portable CPU backend that also runs on
`no_std` and WebAssembly targets.

`burn-ndarray` is the deprecated predecessor of `burn-flex`; it is not related to `burn-cpu`.

## Usage Example

```toml
burn = { version = "0.22", features = ["cpu"] }
```

```rust, ignore
use burn::tensor::{Device, Tensor};

let device = Device::cpu();
let tensor = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
```
