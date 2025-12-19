# Burn Backend Tests

This crate provides a comprehensive suite of tests for Burn backends, covering:

- Tensor operations: [tests/tensor/](./tests/tensor/)
- Autodiff: [tests/autodiff/](./tests/autodiff/)
- (Optional) CubeCL kernels correctness: [tests/cubecl/](./tests/cubecl/)

## Running Tests

The `TestBackend` is selected via feature flags. Use the provided shorthand commands for
convenience:

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

By default, `cargo test` fail-fast across integration test binaries. When one integration test
binary fails, Cargo does not run the remaining test binaries. If you want to run all test binaries
regardless of failures, pass `--no-fail-fast`, for example:

```sh
cargo test-cuda --no-fail-fast
```

## Structure

- `tests/tensor.rs`: Tensor tests
- `tests/autodiff.rs`: Autodiff tests
- `tests/fusion.rs`: Fusion backend tests wrapping tensor and autodiff tests
- `tests/cubecl.rs`: CubeCL kernel tests

Each test module assumes exactly one `FloatElemType`, `IntElemType`, and `TestBackend` in scope.

### Common Modules

- `common/backend.rs`: Backend type definitions
- `common/tensor.rs`: Reusable tensor test suite, split across float, int and bool tensor kinds
- `common/autodiff.rs`: Reusable autodiff test suite, with and without checkpointing

### Test Reusability

This crate uses a pattern of parameterized test modules to run the same tests with different
configurations (backends, dtypes, etc.):

1. **Type aliases define the configuration**: Each test scope declares `FloatElemType`,
   `IntElemType`, and `TestBackend`
1. **`#[path = "..."]` references shared modules**: Points to test files outside the normal module
   hierarchy, e.g. `"common/tensor.rs"`
1. **`include!()` imports test code**: Test modules are included multiple times with different type
   configurations
1. **`use super::*;`** propagates types down the module tree: Each level re-exports parent types so
   deeply nested tests have access to the configured types

For example, `common/tensor.rs` can be included with `FloatElemType = f32` for base tests, then
included again with `FloatElemType = f16` for half-precision tests, running the same test suite
twice with different dtypes.

## Adding New Tests

Add test modules under `tests/tensor/`, `tests/autodiff/`, or `tests/cubecl` respectively. They will
automatically run for all required configurations.

For tensor tests, make sure to add the test to each relevant tensor kind:

- `tensor/bool`: boolean tensor tests
- `tensor/float`: float tensor tests
- `tensor/int`: integer tensor tests

**Guidelines:**

Import types with `use super::*;` at the top of each module and use the types defined in
`common/backend.rs`:

```rust
/// Collection of types used across tests
pub use burn_autodiff::Autodiff;
pub use burn_tensor::Tensor;
pub type TestBackend = ...;

pub type TestTensor<const D: usize> = Tensor<TestBackend, D>;
pub type TestTensorInt<const D: usize> = Tensor<TestBackend, D, burn_tensor::Int>;
pub type TestTensorBool<const D: usize> = Tensor<TestBackend, D, burn_tensor::Bool>;

pub type FloatElem = burn_tensor::ops::FloatElem<TestBackend>;
pub type IntElem = burn_tensor::ops::IntElem<TestBackend>;

pub type TestAutodiffBackend = Autodiff<TestBackend>;
pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;
```

Tests will automatically run with default dtypes and any variants (f16, bf16, etc.) based on the
backend configuration.
