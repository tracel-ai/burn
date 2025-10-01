pub(crate) mod stats;

mod api;
mod data;
mod distribution;
mod element;
mod shape;

pub use api::*;
pub use burn_common::bytes::*;
pub use data::*;
pub use distribution::*;
pub use element::*;
pub use shape::*;

/// The activation module.
pub mod activation;

/// The backend module.
pub mod backend;

/// The container module.
pub mod container;

/// The grid module.
pub mod grid;

/// The indexing module.
pub mod indexing;

pub use indexing::AsIndex;

/// The linalg module.
pub mod linalg;

/// The loss module.
pub mod loss;

/// The burn module.
pub mod module;

/// Operations on tensors module.
pub mod ops;

/// Tensor quantization module.
pub mod quantization;

#[cfg(feature = "std")]
pub use report::*;

#[cfg(feature = "std")]
mod report;

#[cfg(feature = "experimental-named-tensor")]
mod named;

#[cfg(feature = "experimental-named-tensor")]
pub use named::*;

pub use ops::Device; // Re-export device so that it's available from `burn_tensor::Device`.
