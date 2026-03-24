# Burn Backend Tests

This crate provides a comprehensive suite of tests for Burn backends, covering:

- Tensor operations: [tests/tensor/](./tests/tensor/)
- Autodiff: [tests/autodiff/](./tests/autodiff/)
- (Optional) CubeCL kernels correctness: [tests/cubecl/](./tests/cubecl/)
- (Optional) Fusion correctness: [tests/fusion/](./tests/fusion/)

## Running Tests

The test backend device is selected via feature flags. Use the provided shorthand commands for
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

> [!NOTE]  
> CubeCL-based backends are tested with `fusion` by default. If you want to run the tests without
> fusion, just append `-nofuse` to the cargo command. For example:
>
> ```sh
> cargo test-cuda-nofuse
> ```

## Structure

- `tests/tensor.rs`: Tensor tests
- `tests/tensor_f16.rs`: F16 tensor tests
- `tests/autodiff.rs`: Autodiff tests
- `tests/autodiff_f16.rs`: F16 autodiff tests
- `tests/cubecl.rs`: CubeCL kernel tests

### Common Modules

- `common/backend.rs`: Backend type definitions
- `common/tensor.rs`: Reusable tensor test suite, split across float, int and bool tensor kinds
- `common/autodiff.rs`: Reusable autodiff test suite, with and without checkpointing

### Test Reusability

This crate uses a pattern of parameterized test modules to run the same tests with different
configurations:

1. **Element types**: Each test scope declares `FloatElem` and `IntElem`, which determine the
   precision for the current test run
1. **`#[path = "..."]` references shared modules**: Points to test files outside the normal module
   hierarchy, e.g. `"common/tensor.rs"`
1. **`include!()` imports test code**: Test modules are included multiple times with different type
   configurations
1. **`use super::*;`** propagates types down the module tree: Each level re-exports parent types so
   deeply nested tests have access to the configured types

For example, all tensor tests are included in `tensor.rs` to execute with the default element types
(`FloatElem = f32`, `IntElem = i32`):

```rust
#[path = "common/tensor.rs"]
mod tensor;
```

For f16 tests, only the float tensor tests are included to validate behavior with `FloatElem = f16`:

```rust
#[path = "tensor/float/mod.rs"]
mod f16;
```

> [!WARNING]  
> Tests for different data types (e.g., f32 vs f16) must be isolated, as each device's settings are
> global and can only be initialized once per process.

## Adding New Tests

Add test modules under `tests/tensor/`, `tests/autodiff/`, `tests/cubecl` and `tests/fusion`
respectively. They will automatically run for all required configurations.

For tensor tests, make sure to add the test to each relevant tensor kind:

- `tensor/bool`: boolean tensor tests
- `tensor/float`: float tensor tests
- `tensor/int`: integer tensor tests

**Guidelines:**

Import types with `use super::*;` at the top of each module to use the `TestBackend` and `FloatElem`
/ `IntElem` types.

For autodiff tests, always use `AutodiffDevice::new()` to create the device. Autodiff is enabled on
the device itself when using the `Dispatch` test backend, ensuring the device supports automatic
differentiation without modifying the backend type.
