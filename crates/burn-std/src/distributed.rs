use serde::{Deserialize, Serialize};

use crate::Shape;

/// The different ways to execute the reduce operation.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceOperation {
    /// The sum of the values.
    Sum,
    /// The mean of the values.
    Mean,
}

/// Parameter struct for setting up and getting parameters for distributed operations.
#[derive(Clone, Debug)]
pub struct DistributedConfig {
    /// How to execute the all_reduce operation.
    pub all_reduce_op: ReduceOperation,
}

/// How a tensor's shards, one per member of its sharded compute device, make up its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sharding {
    /// Every member holds the whole value.
    Replicated,
    /// Balanced chunks in member order: when the dim does not divide evenly, the first members each
    /// hold one more index along `dim`.
    Split {
        /// The dim the value is split along.
        dim: usize,
    },
    /// The global value is the sum of the shards.
    PartialSum,
}

/// Why a tensor cannot take a sharding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardingError {
    /// Only a tensor on a sharded compute device has a sharding.
    NotSharded,
    /// A quantized tensor splits only into whole quantization blocks, so that no code or scale
    /// changes, and into whole packed words along its packed axis, which a device stores it in.
    QuantizedSplit {
        /// The dim asked for.
        dim: usize,
        /// The tensor's shape.
        shape: Shape,
        /// The sharded compute device's number of members.
        num_members: usize,
    },
    /// The split dim is not one of the tensor's.
    MissingDim {
        /// The dim asked for.
        dim: usize,
        /// The tensor's shape.
        shape: Shape,
    },
    /// Fewer indices than members along the split dim would leave a member empty.
    DimShorterThanMembers {
        /// The dim asked for.
        dim: usize,
        /// The tensor's shape.
        shape: Shape,
        /// The sharded compute device's number of members.
        num_members: usize,
    },
    /// A partial sum is an op's output, never a split of a whole value.
    PartialSumFromWhole {
        /// The tensor's sharding.
        from: Sharding,
    },
}

impl core::fmt::Display for ShardingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSharded => write!(
                f,
                "Only a tensor on a sharded compute device has a sharding"
            ),
            Self::QuantizedSplit {
                dim,
                shape,
                num_members,
            } => write!(
                f,
                "Splitting dim {dim} of a quantized tensor of shape {shape:?} over \
                 {num_members} members would cut its quantization blocks or packed words"
            ),
            Self::MissingDim { dim, shape } => {
                write!(f, "A tensor of shape {shape:?} has no dim {dim} to split")
            }
            Self::DimShorterThanMembers {
                dim,
                shape,
                num_members,
            } => write!(
                f,
                "Dim {dim} of a tensor of shape {shape:?} is shorter than the sharded compute device's \
                 {num_members} members, so a split would leave a member empty"
            ),
            Self::PartialSumFromWhole { from } => {
                write!(
                    f,
                    "A {from:?} tensor is not a partial sum and cannot become one"
                )
            }
        }
    }
}

impl core::error::Error for ShardingError {}
