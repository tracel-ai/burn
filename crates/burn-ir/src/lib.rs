#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Burn's intermediate representation of tensors and tensor operations.
//!
//! Every backend operation has a serializable description here: [`OperationIr`] and its
//! per-kind enums ([`FloatOperationIr`], [`IntOperationIr`], [`BaseOperationIr`], ...) name
//! the operation and the [`TensorIr`]s it reads and writes. [`GraphIr`] groups operations
//! with explicit inputs and outputs, and [`CustomOpIr`] carries operations defined by backend
//! extensions.
//!
//! Describing work as data rather than calls lets it be inspected, optimized and moved before
//! it runs. Kernel fusion (`burn-fusion`), remote execution (`burn-remote`, through
//! `burn-router`) and graph capture (`burn-capture`) are built on it. A backend that
//! implements [`BackendIr`] can execute operations received in this form.
//!
//! Applications do not use this crate directly.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
//! - `tracing`: instrument operations with the `tracing` crate.

extern crate alloc;

mod backend;
mod builder;
mod graph;
mod handle;
mod operation;
mod scalar;
mod tensor;

pub use backend::*;
pub use builder::*;
pub use graph::*;
pub use handle::*;
pub use operation::*;
pub use scalar::*;
pub use tensor::*;
