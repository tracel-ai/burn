# Burn Func

> JAX-style functional transformations for [Burn](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-func.svg)](https://crates.io/crates/burn-func)
[![Documentation](https://docs.rs/burn-func/badge.svg)](https://docs.rs/burn-func)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

This crate provides stateless functional transformations over Burn parameters. A function is
written once against a parameter tree, then differentiated, linearized or vectorized without
threading gradients through module state.

It provides:

- `grad` and `value_and_grad` for reverse-mode gradients, with zero gradients for unused or
  constant leaves and non-destructive extraction for aliased leaves.
- `vjp` and `vjp_once` for vector-Jacobian products, including multi-output cotangents that share
  intermediate nodes.
- `jvp` and `value_and_jvp` for forward-mode Jacobian-vector products, implemented by the
  `ForwardDiff<B>` backend decorator.
- `vmap` and `vmap_with_axes` for vectorized batching, implemented by the `Vmap<B>` backend
  decorator. Full reductions exclude the batch axis, and non-zero input axes are supported.
- `hvp` for Hessian-vector products, with a `Differentiable` trait for functions that must be
  evaluated at nested backend levels.
- `PyTree`, a compile-time parameter model implemented for tensors, tuples, `Vec`, `Option`,
  arrays, scalar metadata, and structs deriving `#[derive(PyTree)]`.

The transformations are generic over `B: Backend` and operate on backend primitives through the
zero-cost `FuncTensor<B, D>` wrapper, so they compose with `Autodiff<B>`, `ForwardDiff<B>` and
`Vmap<B>`. `FuncTensor` supplies the ordinary tensor syntax (`a * b + c`, `x.matmul(w)`,
`x.relu()`) that the transformations need while remaining independent of the backend-erased
`Tensor<D>` API. With the `bridge` feature, `FuncTensor::from_tensor` and
`FuncTensor::into_tensor` convert to and from `Tensor<D>` on dispatch-supported backends.

## PyTree Limitations

Differentiable leaves in a `PyTree` must be float primitives. Non-differentiable metadata fields
marked with `#[pytree(skip)]` must not reference the backend type parameter `B` (e.g., they must be
`B`-independent types such as `usize`, `f32`, or `String`). Integer and boolean tensors (such as indices
or masks) cannot be safely rebound across backends like `Vmap<B>` and must be closed over outside the
parameter tree.

## Forward-Mode (`jvp`) Pushforwards

Forward-mode autodiff (`ForwardDiff<B>`) propagates non-trivial tangents through all elementary
math, activations, reductions, structural operations, convolutions, cumulative scans (`cumprod`,
`cummin`, `cummax`), multi-head attention (`attention`), and grid sampling (`grid_sample_2d`).

## Vectorization (`vmap`) and Data-Dependent Shapes

Operations with dynamic or data-dependent output shapes—specifically `bool_argwhere`, `float_mask_select`,
`int_mask_select`, and `bool_mask_select`—cannot produce a fixed physical batch dimension across lanes.
Invoking these on batched tensors under `Vmap` panics with an actionable error. Masks and selection
logic should either be closed over or computed per-sample.

## Per-Sample Gradients (`vmap(grad(f))`)

Per-sample gradients can be computed efficiently by vectorizing `grad` over batch inputs while keeping
model parameters unbatched:

```rust
let vmap_grads = vmap_with_axes(
    |(w, x, y)| grad(sample_loss)(w),
    vec![VmapAxis::Unbatched, VmapAxis::Batched(0), VmapAxis::Batched(0)],
)(&(w, x_batch, y_batch));
```

See `tests/vmap_tests.rs` for working examples.

## Feature Flags

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- `derive` (default): the `#[derive(PyTree)]` procedural macro.
- `bridge`: conversions between `FuncTensor` and `burn_tensor::Tensor<D>` for dispatch-supported
  backends.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
