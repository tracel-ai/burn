# Burn Backend Extension

> [Burn](https://github.com/tracel-ai/burn) backend extension generation

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-backend-extension.svg)](https://crates.io/crates/burn-backend-extension)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-backend-extension/blob/master/README.md)

`#[backend_extension]` generates runtime dispatch for custom backend operations. Add `Fusion`
to generate a lazy implementation that computes output metadata now and calls the backend later.
Enable Burn's `extension` and `fusion` features plus a CubeCL runtime, such as `cpu` or `wgpu`,
and add `burn-cubecl` as a dependency for the backend implementation.

```rust,ignore
use burn::backend::{Backend, backend_extension, ops::FloatTensorOps, tensor::FloatTensor};
use burn_cubecl::CubeBackend;

#[backend_extension(Cube, Fusion)]
pub trait ScaleOps: Backend {
    #[fusion(dtype = input, shape = input)]
    fn scale(input: FloatTensor<Self>, factor: f32) -> FloatTensor<Self>;
}

impl ScaleOps for CubeBackend {
    fn scale(input: FloatTensor<Self>, factor: f32) -> FloatTensor<Self> {
        Self::float_mul_scalar(input, factor.into())
    }
}
```

Here, `dtype = input` copies the input dtype and `shape = input` copies its shape. For a different
output shape, call a helper shared with the backend implementation, such as `shape = output_shape(input)`.
Use `#[fusion(meta = callable)]` for structured outputs or `#[fusion(default)]` to inherit a trait body.

**For structured outputs, metadata supplies the actual non-tensor return values.** They must match
direct backend execution; the backend's later values are discarded without comparison.

See the [macro documentation in `src/lib.rs`](src/lib.rs)
for the complete metadata contract, supported signatures, validation, and scalar encodings.
The [custom CubeCL kernel tutorial](../../burn-book/src/advanced/backend-extension/custom-cubecl-kernel.md)
shows how to add a kernel and a handwritten backward pass. Fusion generation does not generate gradients
or automatically merge custom kernels with neighboring operations.

## Integration tests

Fusion extension tests live in `burn-core` and require its opt-in `extension-tests` feature plus a
CubeCL runtime. CI includes them in the existing Metal/Fusion test group. To run locally on CPU,
including debug-only validation checks:

```sh
BURN_DEVICE=cpu cargo test -p burn-core --no-default-features --features std,extension-tests,cpu \
  --test backend_extension_fusion --test backend_extension_scalars
```
