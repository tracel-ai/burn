# Burn Intermediate Representation

> [Burn](https://github.com/tracel-ai/burn)'s description of tensors and tensor operations as data

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-ir.svg)](https://crates.io/crates/burn-ir)
[![Documentation](https://docs.rs/burn-ir/badge.svg)](https://docs.rs/burn-ir)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Every backend operation has a serializable description here. `OperationIr` and its per-kind enums
(`FloatOperationIr`, `IntOperationIr`, `BaseOperationIr`, ...) name an operation and the
`TensorIr`s it reads and writes. `GraphIr` groups operations with explicit inputs and outputs, and
`CustomOpIr` carries operations defined by backend extensions.

Describing work as data rather than as calls lets it be inspected, optimized and moved before it
runs. It is the basis of:

- kernel fusion, in [burn-fusion](https://github.com/tracel-ai/burn/tree/main/crates/burn-fusion);
- remote execution, in [burn-remote](https://github.com/tracel-ai/burn/tree/main/crates/burn-remote) through [burn-router](https://github.com/tracel-ai/burn/tree/main/crates/burn-router);
- graph capture, in [burn-capture](https://github.com/tracel-ai/burn/tree/main/crates/burn-capture).

A backend that implements `BackendIr` can execute operations received in this form. Applications
do not use this crate directly.

## Feature Flags

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- `tracing`: instrument operations with the `tracing` crate.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
