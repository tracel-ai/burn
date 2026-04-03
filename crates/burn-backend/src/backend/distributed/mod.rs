//! Types and helpers for inter-device operations.

pub(crate) mod all_reduce;
pub(crate) mod api;
pub(crate) mod client;
mod ops;
pub(crate) mod server;

pub use api::*;
pub use ops::*;

// TODO: should be removed I think!
// instead of having a new `DistributedParamId`, we can keep `ParamId` only since it maps to the param id.
// (in `ModuleSharder` the map_float impl simply calls `DistributedParamId::from(param_id)`).
// and we don't need to query the tensor to get the param id. Simply, check *if* `tensor.is_distributed()`
// and then afterwards , if t was distributed we can recover the state here since we have the param id.
use serde::{Deserialize, Serialize};

/// A unique identifier for a parameter distributed across multiple devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DistributedParamId(u64);

impl From<u64> for DistributedParamId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

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
