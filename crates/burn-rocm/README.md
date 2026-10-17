# Burn ROCm Backend

> [Burn](https://github.com/tracel-ai/burn) ROCm backend for AMD GPUs

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-rocm.svg)](https://crates.io/crates/burn-rocm)
[![Documentation](https://docs.rs/burn-rocm/badge.svg)](https://docs.rs/burn-rocm)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

This crate provides the ROCm (HIP) runtime of Burn's [CubeCL backend](https://github.com/tracel-ai/burn/tree/main/crates/burn-cubecl), using
[CubeCL](https://github.com/tracel-ai/cubecl) and
[cubecl-hip-sys](https://github.com/tracel-ai/cubecl-hip-sys).

## Usage Example

For application code, enable Burn's `rocm` feature and select the device at runtime:

```toml
burn = { version = "0.22", features = ["rocm"] }
```

```rust,ignore
use burn::tensor::{Device, Tensor};

let device = Device::rocm(0);
let input = Tensor::<2>::ones([2, 3], &device);
let output = input + 1.0;
```

For training, enable `autodiff` (also enabled by `train`) and use `device.autodiff()` before
initializing model parameters and inputs.

## Requirements

- Linux with an AMD GPU supported by ROCm.
- A [ROCm installation](https://rocm.docs.amd.com/) at run time. The HIP libraries are loaded when
  the first device is created, so building does not require ROCm. Burn 0.22 uses the bindings for
  HIP 60850 (ROCm 7.14); see [cubecl-hip-sys](https://github.com/tracel-ai/cubecl-hip-sys) for how HIP
  and ROCm versions map.
- Set `ROCM_PATH` or `HIP_PATH` when ROCm is not installed in its default location (often
  `/opt/rocm`).

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
