#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Burn's tensor API.
//!
//! [`Tensor<D, K>`](Tensor) is a tensor of rank `D` and kind `K` ([`Float`] by default, [`Int`]
//! or [`Bool`]). Its [`Device`] decides which backend runs it, at runtime: the same `Tensor<2>`
//! can live on CUDA, wgpu or the CPU, and code that uses tensors has no backend type parameter.
//! Applications use this crate through `burn::tensor`.
//!
//! ```rust,no_run
//! use burn_tensor::{Device, Tensor, s};
//!
//! let device = Device::default();
//! let x = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
//! let y = x.clone().matmul(x.transpose()).exp();
//! let first_row = y.slice(s![0..1, ..]);
//! ```
//!
//! - [`Device`]: backend selection, configuration and enumeration. A backend's constructor
//!   (`Device::cuda`, `Device::wgpu`, `Device::flex`, ...) exists when its feature is enabled.
//! - [`TensorData`], [`DType`] and [`Shape`]: tensor contents and metadata, independent of any
//!   backend.
//! - [`activation`], [`loss`] and [`module`]: functional forms of activations, losses and neural
//!   network operations such as convolution and pooling.
//! - [`quantization`], [`grid`] and [`distributed`]: quantized tensors, grid sampling and
//!   collective operations across devices.
//! - `einsum!`, `assert_shape!` and [`s!`]: macros for Einstein summation, shape checks and
//!   slicing.
//!
//! With the `autodiff` feature, create tensors on a `device.autodiff()` device and mark source
//! leaves with [`Tensor::require_grad`] to record gradients.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
//! - Backends: `cuda`, `rocm`, `wgpu`, `metal`, `vulkan`, `webgpu`, `cpu`, and `flex`.
//! - `autodiff`, `fusion`: backend decorators.
//! - `remote`, `remote-server`, `remote-websocket`: devices hosted by another machine.
//! - `capture`: record operation graphs instead of executing them.
//! - `extension`: access to backend primitives for backend extensions.
//! - `tracing`: instrument operations with the `tracing` crate.
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
mod tensor;

pub(crate) use tensor::check::macros::check;
pub use tensor::*;

mod einsum_macros;
mod shape_macros;
#[doc(hidden)]
pub use burn_derive::{__assert_shape, __debug_assert_shape, __einsum};

// Re-exported types
#[cfg(feature = "autodiff")]
pub use burn_dispatch::GradientCheckpointingStrategy;
pub use burn_std::{
    AllocationProperty, Bytes, bf16, f16, flex32,
    reader::{read_sync, try_read_sync},
    stream::StreamId,
};

mod device;
pub use device::*;

/// Configure a wgpu runtime or share an application's existing wgpu setup.
#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "remote")]
pub mod remote;
#[cfg(feature = "remote-server")]
pub mod server;

pub(crate) use burn_backend::TensorPrimitive;
