#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! JAX-style functional transformations for Burn.
//!
//! This crate operates on backend primitives through [`FuncTensor`], a zero-cost wrapper that
//! restores the tensor ergonomics needed to write JAX-like expressions while remaining generic
//! over any [`Backend`](burn_backend::Backend).
//!
//! The transformations are pure functions:
//!
//! - [`grad()`] and [`value_and_grad()`] compute reverse-mode gradients.
//! - [`vjp()`] computes vector-Jacobian products.
//! - [`jvp()`] and [`value_and_jvp()`] compute forward-mode Jacobian-vector products.
//! - [`vmap()`] vectorizes a function over a batch axis.
//!
//! Parameters are traversed with [`PyTree`]. Implementations are provided for [`FuncTensor`],
//! tuples, `Vec`, `Option`, arrays, scalar metadata, and structs deriving [`PyTree`].
//!
//! # PyTree and Leaf Types
//!
//! Differentiable leaves are float primitives only. Fields marked `#[pytree(skip)]` must be
//! independent of the backend parameter `B`. Close over integer and boolean tensors (indices,
//! masks) outside the parameter tree.
//!
//! # `FuncTensor` vs `Tensor`
//!
//! The functionals operate on [`FuncTensor<B, D>`], a zero-cost wrapper over backend float
//! primitives (`B::FloatTensorPrimitive`), rather than `burn_tensor::Tensor`. When the `bridge`
//! feature is enabled, `FuncTensor::from_tensor` and `FuncTensor::into_tensor` convert
//! between `FuncTensor` and `Tensor<Dispatch>`.
//!
//! # Forward-Mode (`jvp`) Pushforwards
//!
//! Forward-mode automatic differentiation propagates tangents through elementary math,
//! activations, convolutions, cumulative ops ([`cumprod`](FuncTensor::cumprod),
//! [`cummin`](FuncTensor::cummin), [`cummax`](FuncTensor::cummax)), multi-head [`attention`](FuncTensor::attention),
//! and [`grid_sample_2d`](FuncTensor::grid_sample_2d).
//!
//! # `vmap` and Data-Dependent Shapes
//!
//! Operations with data-dependent output shapes (`bool_argwhere`, `float_mask_select`,
//! `int_mask_select`, `bool_mask_select`) cannot be vectorized with a uniform batch dimension
//! and panic when invoked on batched tensors under [`vmap()`].
//!
//! # Per-Sample Gradients (`vmap(grad(f))`)
//!
//! Per-sample weight gradients can be computed efficiently by vectorizing `grad` over inputs
//! while keeping parameters unbatched:
//!
//! ```ignore
//! let per_sample_grads = vmap_with_axes(
//!     |(w, x, y)| grad(sample_loss)(w),
//!     vec![VmapAxis::Unbatched, VmapAxis::Batched(0), VmapAxis::Batched(0)],
//! )(&(weights, batch_x, batch_y));
//! ```
//!
//! # Example
//!
//! ```ignore
//! use burn_func::{FuncTensor, grad};
//!
//! let x = FuncTensor::<MyBackend, 1>::from_data([2.0], &device);
//! let f = |x: &FuncTensor<MyBackend, 1>| (x.clone() * 3.0).sum();
//! let g = grad(f);
//! let df = g(&x);
//! assert_eq!(df.into_data_sync().to_vec::<f32>(), vec![3.0]);
//! ```

extern crate alloc;

/// Re-exported so the `PyTree` derive can name backend types without requiring users to add
/// `burn-backend` as a direct dependency.
pub use burn_backend;

/// Forward-mode functional transformations.
pub mod forward_diff;
/// Reverse-mode functional transformations.
pub mod grad;
/// Compile-time parameter traversal and backend rebinding.
pub mod pytree;
/// Ergonomic zero-cost tensor wrapper over backend primitives.
pub mod tensor;
/// Vectorized batching.
pub mod vmap;

pub use forward_diff::{DualFloatTensor, ForwardDiff, jvp, value_and_jvp};
pub use grad::{Differentiable, grad, hvp, hvp_fn, value_and_grad, vjp, vjp_once};
pub use pytree::PyTree;
pub use tensor::FuncTensor;
pub use vmap::{BatchedPrimitive, Vmap, VmapAxis, vmap, vmap_with_axes};

#[cfg(feature = "derive")]
pub use burn_derive::PyTree;
