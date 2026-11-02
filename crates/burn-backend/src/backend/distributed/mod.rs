//! Types and helpers for inter-device operations.

#[cfg(feature = "std")]
pub(crate) mod api;
#[cfg(feature = "std")]
pub(crate) mod client;
mod ops;
#[cfg(feature = "std")]
pub(crate) mod server;

#[cfg(feature = "std")]
pub use api::*;
pub use ops::*;

pub use burn_std::distributed::*;
/// A unique identifier for a parameter distributed across multiple devices.
pub type DistributedParamId = burn_std::id::ParamId;

/// Parameters for a tensor that is sharded across multiple devices.
#[derive(Debug, Clone)]
pub struct DistributedParams {
    /// The tensor's [DistributedParamId].
    pub param_id: DistributedParamId,
}
