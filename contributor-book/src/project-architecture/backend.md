# Backend

The low-level `Backend` trait defines the implementation contract for tensor operations. Its
`BackendTypes` supertrait supplies device, float, integer, boolean, quantized tensor, and captured
graph primitive types. Backend implementations and decorators remain generic where useful;
application tensors and modules select them at runtime through `Device`.

See [Tensor](./tensor.md) for the Tensor → Bridge → Dispatch → Backend path.

## Element types

Precision is a runtime property. `BackendTypes` no longer has associated `FloatElem` or `IntElem`
types. Tensor primitives expose their dtype through metadata, and operations accept dtype or scalar
information as needed.

Users can configure default dtypes with `Device::configure(DeviceConfig::default()...)` before
creating tensors, request an explicit dtype when creating a tensor, or call `tensor.cast(...)`.
Device settings are initialized once; creating the first tensor uses the backend's defaults if
configuration has not happened earlier. A backend reports which dtypes it supports through
`supports_dtype`. See the Burn Book's
[Device Settings](https://burn.dev/books/burn/building-blocks/backend.html#device-settings) section
for configuration and per-tensor dtype examples.

Do not assume every backend supports every precision. Backend kernels should handle the dtypes they
advertise, and shared tests use the test target's `FloatElem` and `IntElem` aliases.

## Operations

Backend operations are associated functions taking and returning primitives. Backends may enqueue
work asynchronously and reuse uniquely owned storage. The public tensor API handles user-facing
shape checks, while the bridge and dispatch layers route to the appropriate primitive operations.

CubeCL runtimes share `burn_cubecl::CubeBackend`; the device selects CUDA, ROCm, WGPU, or CPU.
`burn_cubecl::Cube` wraps that backend with `Fusion` when the fusion feature is enabled. Runtime
facade names such as `Cuda` and `Wgpu` are aliases of `Cube`, not distinct extension selectors.

## Autodiff

`burn_autodiff::Autodiff<B, C>` decorates a backend with first-order reverse-mode differentiation,
where `C` is a checkpoint strategy. The low-level `AutodiffBackend` trait defines backward and
gradient access. At the public API, autodiff is a runtime context carried by tensors and supplied as
a creation default by devices.

Enabling autodiff does not automatically track every input. Source leaves use `require_grad()`;
operations depending on tracked inputs record backward steps. Backward consumes reachable steps,
including steps shared by cloned handles. Repeated backward through consumed intermediates is
rejected; parameter leaves can be reused in fresh forwards.

`#[backend_extension(Autodiff, Cube)]` generates runtime routing for an extension. The author still
supplies its implementation for `Autodiff<B, C>` (a composition of differentiable operations or a
custom backward pass). See the
[extension guide](https://burn.dev/books/burn/advanced/backend-extension/).
