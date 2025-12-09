# Burn Backend Tensor Tests

This crate provides a comprehensive suite of tests for Burn backends, covering:

- Tensor operations: [src/tensor/](./src/tensor/)
- Autodiff: [src/autodiff/](./src/autodiff/)
- (Optional) CubeCL kernels correctness: [src/cubecl/](./src/cubecl/)

Additional configuration:

- Autodiff tests are repeated with checkpointing.
- Tensors operations and autodiff tests (including checkpointing) are repeated for
  `Fusion<CubeBackend<...>>`

The `TestBackend` is selected via feature flags. Use the provided shorthand commands for convenience:

```sh
# Cpu
cargo test-cpu
# Cuda
cargo test-cuda
# Rocm
cargo test-rocm
# Wgpu / WebGpu
cargo test-wgpu
# Vulkan
cargo test-vulkan
# Metal
cargo test-metal
# Router
cargo test-router

# Candle
cargo test-candle
# NdArray
cargo test-ndarray
# LibTorch
cargo test-tch
```
