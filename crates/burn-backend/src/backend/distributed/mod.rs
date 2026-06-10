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

use crate::{Backend, tensor::FloatTensor};

/// Parameters for a tensor that is sharded across multiple devices.
#[derive(Debug, Clone)]
pub struct DistributedParams {
    /// The tensor's [DistributedParamId].
    pub param_id: DistributedParamId,
}

/// A tensor handle used for a collective operation, that is not yet valid for use.
/// We must ensure collective operations are completed before accessing the underlying data.
#[derive(new, Clone)]
pub struct CollectiveTensor<B: Backend> {
    handle: FloatTensor<B>,
}

impl<B: Backend> CollectiveTensor<B> {
    /// Synchronizes the collective operation and returns a valid tensor handle.
    pub fn resolve(self) -> FloatTensor<B> {
        B::sync_collective(&B::float_device(&self.handle));
        self.handle
    }

    /// Returns the tensor handle without synchronizing.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `sync_collective()` is called before
    /// the returned handle is used in any computation.
    pub unsafe fn assume_resolved(self) -> FloatTensor<B> {
        self.handle
    }
}
