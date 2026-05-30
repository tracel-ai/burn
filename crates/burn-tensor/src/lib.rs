#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This library provides the core abstractions required to run tensor operations with Burn.
//! `Tensor`s are generic over the backend to allow users to perform operations using different `Backend` implementations.
//! Burn's tensors also support auto-differentiation thanks to the `AutodiffBackend` trait.
//!
//! # Note for contributors: `*_impl` helpers
//!
//! Throughout this crate (e.g. in `tensor::api::float`, `tensor::api::int`,
//! `tensor::api::bool`, `tensor::api::cast`, and `tensor::activation`), public
//! generic methods on `Tensor<D, K>` that need to call into `burn_dispatch` are
//! routed through small non-generic helper functions named `*_impl`, grouped
//! together at the bottom of each file under a banner like:
//!
//! ```text
//! // =====================================================================
//! // Non-generic implementation helpers (outlined from the generic API).
//! // =====================================================================
//! ```
//!
//! These helpers take and return only `BridgeTensor` (a type-erased blob — no
//! `DispatchTensor` or other `burn_dispatch` types appear in their
//! signatures). Because the helpers are not generic over `D`, they are
//! compiled once, and the MIR of the public generic methods does not mention
//! any `burn_dispatch` types. Downstream crates that monomorphize the public
//! generic API therefore never have to resolve the cubecl-backed type tree,
//! which drastically cuts compile times for user code.
//!
//! When adding a new public method that calls a `Dispatch::*` op, follow the
//! same pattern: keep the generic method body thin and forward to a
//! non-generic `*_impl` helper alongside the existing ones.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod bridge;

mod split;
mod tensor;

pub use split::*;
pub(crate) use tensor::check::macros::check;
pub use tensor::*;

// Re-exported types
pub use burn_std::{
    AllocationProperty, Bytes, bf16, f16,
    reader::{read_sync, try_read_sync},
    stream_id::StreamId,
};

mod device;
pub use device::*;

#[cfg(feature = "server")]
pub mod server;

pub(crate) use burn_backend::TensorPrimitive;
