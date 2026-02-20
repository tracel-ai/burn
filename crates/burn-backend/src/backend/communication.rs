use serde::{Deserialize, Serialize};

/// The different ways to execute the reduce operation.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceOperation {
    /// The sum of the values.
    Sum,
    /// The mean of the values.
    Mean,
}

/// All reduce can be implemented with different algorithms, which all have the same result.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum AllReduceStrategy {
    /// One device is the "central". The other devices, "peripherals", send their tensors to the
    /// central. The central does the reduction, and sends the result back to each peripheral.  
    Centralized,

    /// Devices are organized in a tree structure (with a given arity). Each node reduces its
    /// children's tensors with its own, and sends the result to its parent. Leaf nodes will
    /// simply send their tensors to their parents.
    /// When the root node calculates the result, it is propagated down the tree.
    Tree(u32),

    /// Devices are organized in a ring. The tensors are split into N slices, where N is the
    /// number of devices participating. The slices are progressively sent around the ring until
    /// every device has one fully reduced slice of the tensor. Then, the resulting slices are sent
    /// around until every device has the full result.
    /// See `ring.rs` for details.
    Ring,
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

/// A unique identifier for a parameter in a burn module.
/// The same parameter in a clone of the module should have the same id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModuleParamId(u64);

impl From<u64> for ModuleParamId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Parameters for a tensor that is sharded across multiple devices.
#[derive(Debug, Clone)]
pub struct ShardedParams {
    /// The [PeerId](PeerId) of the device.
    pub peer_id: PeerId,
    /// The reduce operation.
    pub op: ReduceOperation,
    // TODO: Should we be able to mark a tensor as sharded if it has no param ID.
    /// The tensor's parameter id if it's in a burn module.
    pub param_id: Option<ModuleParamId>,
}
