# Burn Backend Dispatch

> Runtime backend selection for [Burn](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-dispatch.svg)](https://crates.io/crates/burn-dispatch)
[![Documentation](https://docs.rs/burn-dispatch/badge.svg)](https://docs.rs/burn-dispatch)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

`Dispatch` is the backend behind every `burn::tensor::Tensor`. It holds the tensor primitives of
each backend compiled into the application and routes every operation to the backend that owns
its tensors, so model code has no backend type parameter and one program can use several
backends side by side.

Operations follow `Tensor -> BridgeTensor -> DispatchTensor -> backend primitive`. Dispatch also
carries each tensor's autodiff and gradient-checkpointing context.

Applications do not use this crate directly: Burn's backend features (`cuda`, `wgpu`, `flex`, ...)
enable the matching dispatch features, and a `Device` constructor picks the backend at runtime.

## Backends

| Variant    | Features                                                   | Backend                                                       |
| ---------- | ---------------------------------------------------------- | ------------------------------------------------------------- |
| `Cube`     | `cpu`, `cuda`, `metal`, `rocm`, `vulkan`, `webgpu`, `wgpu` | [burn-cubecl](https://github.com/tracel-ai/burn/tree/main/crates/burn-cubecl), every CubeCL runtime           |
| `Flex`     | `flex`                                                     | [burn-flex](https://github.com/tracel-ai/burn/tree/main/crates/burn-flex), pure Rust CPU                      |
| `Remote`   | `remote`                                                   | [burn-remote](https://github.com/tracel-ai/burn/tree/main/crates/burn-remote), devices on another server      |
| `Capture`  | `capture`                                                  | [burn-capture](https://github.com/tracel-ai/burn/tree/main/crates/burn-capture), records instead of executing |
| `NdArray`  | `ndarray`                                                  | [burn-ndarray](https://github.com/tracel-ai/burn/tree/main/crates/burn-ndarray), deprecated                   |
| `LibTorch` | `tch`                                                      | [burn-tch](https://github.com/tracel-ai/burn/tree/main/crates/burn-tch), deprecated                           |

`autodiff` wraps any of them in [burn-autodiff](https://github.com/tracel-ai/burn/tree/main/crates/burn-autodiff), and `fusion` turns on kernel
fusion for the CubeCL and remote backends. Features combine freely.

Backend extensions add operations to dispatch with the `#[backend_extension]` macro from
[burn-backend-extension](https://github.com/tracel-ai/burn/tree/main/crates/burn-backend-extension).

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
