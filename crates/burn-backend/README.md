# Burn Backend

> The contract between [Burn](https://github.com/tracel-ai/burn)'s tensor API and the backends that execute it

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-backend.svg)](https://crates.io/crates/burn-backend)
[![Documentation](https://docs.rs/burn-backend/badge.svg)](https://docs.rs/burn-backend)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Application code does not use this crate directly: it works with `burn::tensor::Tensor` and
`Device`, and Burn dispatches each operation to a backend at runtime. This crate is for code below
that boundary: backend implementations, backend decorators such as autodiff and fusion, and
backend extensions.

It defines:

- `BackendTypes`, which names a backend's tensor primitives and device type.
- `Backend` and the operation traits in `ops`, every tensor operation a backend implements.
  `AutodiffBackend` adds gradient support.
- `DeviceOps`, a backend device and its default dtypes.
- `TensorData`, `DType`, `Shape` and the element traits, which describe tensor data
  independently of any backend.

Operations report device failures as `ExecutionError` instead of panicking.

## Feature Flags

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- `cubecl`: conversions between Burn and [CubeCL](https://github.com/tracel-ai/cubecl) types.
- `cubecl-device` and `cubecl-<runtime>`: implement `DeviceOps` for `cubecl::Device`.
- `tracing`: instrument operations with the `tracing` crate.

See the [contributor book](https://burn.dev/books/contributor/project-architecture/backend.html)
for how backends fit into Burn's architecture.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
