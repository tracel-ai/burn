pub(crate) mod stats;

mod api;
mod data;
mod element;
mod shape;

pub use api::*;
pub use data::*;
pub use element::*;
pub use shape::*;

/// The activation module.
pub mod activation;

/// The backend module.
pub mod backend;

/// The container module.
pub mod container;

/// The loss module.
pub mod loss;

/// The burn module.
pub mod module;

/// Operations on tensors module.
pub mod ops;

#[cfg(feature = "experimental-named-tensor")]
mod named;
#[cfg(feature = "experimental-named-tensor")]
pub use named::*;
