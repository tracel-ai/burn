use serde::{Deserialize, Serialize};

use alloc::vec::Vec;

use crate::{DType, Shape};

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

/// One tensor of a sharded compute device as its shards: one per member, in member order, and how
/// they make up its value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shards<T> {
    /// Member `i`'s shard at position `i`.
    pub shards: Vec<T>,
    /// How the shards make up the value.
    pub sharding: Sharding,
}

impl<T> Shards<T> {
    /// The same shards, each one converted by `f`.
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> Shards<U> {
        Shards {
            shards: self.shards.into_iter().map(f).collect(),
            sharding: self.sharding,
        }
    }
}

/// Why a tensor cannot take a sharding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardingError {
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
    /// Quantized shards join into one tensor only when they share the scales it keeps once: a
    /// per-tensor scale, or a two-level scheme's global one.
    QuantizedScales {
        /// The first shard whose scale differs from shard 0's.
        shard: usize,
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
            Self::QuantizedScales { shard } => write!(
                f,
                "Quantized shard {shard} has a different per-tensor or global scale from shard 0, \
                 so the shards do not join into one tensor"
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

/// Why shards do not make up one tensor of a sharded compute device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardsError {
    /// Only a sharded compute device takes shards, and only of its members' backend.
    NotSharded,
    /// A sharded compute device takes one shard per member.
    Count {
        /// How many shards were given.
        shards: usize,
        /// The sharded compute device's number of members.
        num_members: usize,
    },
    /// Shard `i` lives on member `i`'s device.
    Device {
        /// The shard, by its position.
        shard: usize,
    },
    /// The shards of one tensor share its dtype, a quantized tensor's scheme included.
    DType {
        /// The shard, by its position.
        shard: usize,
        /// The first shard's dtype.
        expected: DType,
        /// This shard's dtype.
        found: DType,
    },
    /// A shard's shape is not its member's part of the tensor: the whole value when replicated or
    /// a partial sum, and its balanced chunk of the split dim when split.
    Shape {
        /// The shard, by its position.
        shard: usize,
        /// The shape its member's part has.
        expected: Shape,
        /// This shard's shape.
        found: Shape,
    },
    /// Only a float or int tensor that is not quantized sums.
    PartialSumOf {
        /// The shards' dtype.
        dtype: DType,
    },
    /// The tensor the shards make up cannot take the sharding.
    Sharding(ShardingError),
}

impl core::fmt::Display for ShardsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSharded => write!(
                f,
                "Only a sharded compute device takes shards, each on a device of its members' \
                 backend"
            ),
            Self::Count {
                shards,
                num_members,
            } => write!(
                f,
                "{shards} shards for a sharded compute device of {num_members} members, which \
                 takes one per member"
            ),
            Self::Device { shard } => {
                write!(f, "Shard {shard} is not on member {shard}'s device")
            }
            Self::DType {
                shard,
                expected,
                found,
            } => write!(
                f,
                "Shard {shard} is {found:?}, and the first shard {expected:?}"
            ),
            Self::Shape {
                shard,
                expected,
                found,
            } => write!(
                f,
                "Shard {shard} has shape {found:?}, and its member's part is {expected:?}"
            ),
            Self::PartialSumOf { dtype } => {
                write!(f, "A {dtype:?} tensor is not a partial sum")
            }
            Self::Sharding(error) => error.fmt(f),
        }
    }
}

impl core::error::Error for ShardsError {}

impl From<ShardingError> for ShardsError {
    fn from(error: ShardingError) -> Self {
        Self::Sharding(error)
    }
}
