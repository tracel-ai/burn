# Burn Candle Backend

> **Deprecated:** This crate is deprecated as of `0.21.0` and will be removed in a future release.
> Please migrate to one of the actively maintained backends:
> - **CubeCL backends** (CUDA, ROCm, Vulkan, Metal, WebGPU) for GPU acceleration
> - **NdArray** for portable CPU execution
> - **LibTorch** (`burn-tch`) for a mature CPU/GPU backend

This crate provides a backend for [Burn](https://github.com/tracel-ai/burn) based on the [Candle](https://github.com/huggingface/candle) framework.

## Feature Flags

- `cuda` - Cuda GPU device (NVIDIA only)
- `accelerate` - Accelerate framework (macOS only)
