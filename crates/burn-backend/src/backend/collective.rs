use serde::{Deserialize, Serialize};

/// The different ways to execute the reduce operation.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceOperation {
    /// The sum of the values.
    Sum,
    /// The mean of the values.
    Mean,
}

/// A unique identifier for a peer in the context of collective operations.
/// They must be unique, even in multi-node contexts.
///
/// This is like the rank in NCCL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PeerId(u32);

impl core::fmt::Display for PeerId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PeerId({})", self.0)
    }
}

impl From<u32> for PeerId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<i32> for PeerId {
    fn from(value: i32) -> Self {
        Self(value as u32)
    }
}

impl From<usize> for PeerId {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

/// Parameters for a tensor that is sharded across multiple devices.
#[derive(Debug, Clone)]
pub struct ShardedParams {
    /// The [PeerId](PeerId) of the device.
    pub peer_id: PeerId,
    /// The reduce operation.
    pub op: ReduceOperation,
}
