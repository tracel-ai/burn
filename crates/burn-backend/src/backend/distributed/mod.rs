//! Types and helpers for inter-device operations.

pub(crate) mod api;
pub(crate) mod client;
mod ops;
pub(crate) mod server;

pub use api::*;
pub use ops::*;

use serde::{Deserialize, Serialize};

/// A unique identifier for a parameter distributed across multiple devices.
pub type DistributedParamId = burn_std::id::ParamId;

use crate::tensor::FloatTensor;

/// Parameters for a tensor that is sharded across multiple devices.
#[derive(Debug, Clone)]
pub struct DistributedParams {
    /// The tensor's [DistributedParamId].
    pub param_id: DistributedParamId,
}

/// The different ways to execute the reduce operation.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceOperation {
    /// The sum of the values.
    Sum,
    /// The mean of the values.
    Mean,
}

/// Parameter struct for setting up and getting parameters for distributed operations.
#[derive(Clone)]
pub struct DistributedConfig {
    /// How to execute the all_reduce operation.
    pub all_reduce_op: ReduceOperation,
}

/// A tensor handle used for a collective operation, that is not yet valid for use.
/// We must ensure collective operations are completed before accessing the underlying data.
#[derive(new, Clone)]
pub struct CollectiveTensor<B: DistributedBackend> {
    handle: FloatTensor<B>,
}

impl<B: DistributedBackend> CollectiveTensor<B> {
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
