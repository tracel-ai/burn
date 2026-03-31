use serde::{Deserialize, Serialize};

// TODO: this can probably be removed, it's only ever derived from `ParamId` and then used in distributed setting?

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
