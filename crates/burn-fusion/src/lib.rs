#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Fusion
//!
//! Kernel fusion as a backend decorator.
//!
//! [`Fusion`] wraps a backend that implements [`FusionBackend`] and defers its operations
//! instead of running each one immediately. Queued operations are grouped by
//! [`OperationFuser`]s into [`Optimization`]s, so several element-wise operations can run as a
//! single kernel and intermediate tensors never reach device memory.
//!
//! Applications do not use this crate directly. The CubeCL backends (CUDA, ROCm, Metal,
//! Vulkan, WebGPU, wgpu and the CubeCL CPU runtime) enable it by default. Backend authors implement [`FusionBackend`] and
//! [`FusionRuntime`] to opt in; [`custom`] lets backend extensions register their own fused
//! operations.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support.
//! - `memory-checks`: check the fusion runtime for leaked tensors (for tests).
//! - `tracing`: instrument operations with the `tracing` crate.

#[macro_use]
extern crate derive_new;

/// Client module exposing types to communicate with the fusion server.
pub mod client;
/// Stream module exposing all tensor operations that can be optimized.
pub mod stream;

/// Search module for stream optimizations.
pub(crate) mod search;

mod backend;
mod op;
mod ops;
mod server;
mod tensor;

pub mod observer;

/// Test-only introspection into fusion runtime behavior — see
/// [`inspect::FusionInspector`].
#[cfg(feature = "test-util")]
pub mod inspect;

pub use op::UnfusedOp;

/// The error an [operation](stream::Operation) reports when it cannot run.
///
/// Re-exported because the trait names it, so anything implementing an
/// operation can reach it through this crate rather than taking a dependency
/// of its own.
pub use burn_backend::ExecutionError;
pub(crate) use server::*;

pub use backend::*;
pub use ops::NoOp;
pub use tensor::*;

/// Types used to define and register custom Fusion operations.
///
/// Backend extension crates can use this module without depending directly on
/// `burn-ir` or coordinating its version with `burn-fusion`.
pub mod custom;
