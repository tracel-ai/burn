use serde::{Deserialize, Serialize};

use alloc::{string::String, vec::Vec};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Sharding {
    /// Every member holds the whole value.
    Replicated,
    /// Balanced chunks in member order: when the dim does not divide evenly, the first members each
    /// hold one more index along `dim`.
    Split {
        /// The dim the value is split along.
        dim: usize,
    },
    /// Balanced chunks, one per group of members: the members form `groups` contiguous groups,
    /// and every member of group `i` holds chunk `i` along `dim`, cut as [`Split`](Self::Split)
    /// cuts one over `groups` members. Which members pair up is chosen by their order on the
    /// sharded compute device.
    ReplicatedSplit {
        /// The dim the value is split along.
        dim: usize,
        /// How many chunks the value is cut into. It divides the number of members, and is
        /// neither one, which is [`Replicated`](Self::Replicated), nor every member, which is
        /// [`Split`](Self::Split).
        groups: usize,
    },
    /// The global value is the sum of the shards.
    PartialSum,
}

impl Sharding {
    /// The dim a split cuts, or `None` when the value is not split.
    pub fn split_dim(self) -> Option<usize> {
        match self {
            Self::Split { dim } | Self::ReplicatedSplit { dim, .. } => Some(dim),
            Self::Replicated | Self::PartialSum => None,
        }
    }

    /// The same sharding split along `dim` instead, or unchanged when it is not split.
    pub fn on_dim(self, dim: usize) -> Self {
        match self {
            Self::Split { .. } => Self::Split { dim },
            Self::ReplicatedSplit { groups, .. } => Self::ReplicatedSplit { dim, groups },
            sharding => sharding,
        }
    }

    /// How many chunks a split cuts the value into over `num_members` members, or `None` when it
    /// is not split.
    pub fn num_chunks(self, num_members: usize) -> Option<usize> {
        match self {
            Self::Split { .. } => Some(num_members),
            Self::ReplicatedSplit { groups, .. } => Some(groups),
            Self::Replicated | Self::PartialSum => None,
        }
    }
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

/// The times one operation moved an input of a sharded compute device between members, from
/// one sharding to another.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReshardingTally {
    /// The operation that asked for its input in the sharding `to`, such as `BaseFloat::Select`.
    pub operation: String,
    /// The input's sharding before.
    pub from: Sharding,
    /// The sharding the operation asked for.
    pub to: Sharding,
    /// How many times.
    pub count: u64,
    /// The size of the whole inputs as stored, summed: a quantized tensor's scales included.
    pub bytes: u64,
}

impl ReshardingTally {
    /// What was added to the tally since `earlier`, a tally of the same operation and shardings.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            count: self.count - earlier.count,
            bytes: self.bytes - earlier.bytes,
            ..self.clone()
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
        /// The chunks the split cuts: the members, or a replicated split's groups.
        chunks: usize,
    },
    /// The split dim is not one of the tensor's.
    MissingDim {
        /// The dim asked for.
        dim: usize,
        /// The tensor's shape.
        shape: Shape,
    },
    /// Fewer indices than chunks along the split dim would leave a member empty.
    DimShorterThanChunks {
        /// The dim asked for.
        dim: usize,
        /// The tensor's shape.
        shape: Shape,
        /// The chunks the split cuts: the members, or a replicated split's groups.
        chunks: usize,
    },
    /// A replicated split's groups must divide the members, and be neither one nor every member.
    InvalidGroups {
        /// The groups asked for.
        groups: usize,
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
    /// The sharded compute device was made with strict gathers, which refuse to make a split
    /// tensor whole on every member.
    SplitGathered {
        /// The tensor's sharding.
        from: Sharding,
        /// The sharding asked for.
        to: Sharding,
    },
}

impl core::fmt::Display for ShardingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::QuantizedSplit { dim, shape, chunks } => write!(
                f,
                "Splitting dim {dim} of a quantized tensor of shape {shape:?} into {chunks} \
                 chunks would cut its quantization blocks or packed words"
            ),
            Self::MissingDim { dim, shape } => {
                write!(f, "A tensor of shape {shape:?} has no dim {dim} to split")
            }
            Self::DimShorterThanChunks { dim, shape, chunks } => write!(
                f,
                "Dim {dim} of a tensor of shape {shape:?} is shorter than the {chunks} chunks \
                 a split cuts it into, so a member would be left empty"
            ),
            Self::InvalidGroups {
                groups,
                num_members,
            } => write!(
                f,
                "{groups} groups do not make a replicated split over {num_members} members: \
                 they must divide the members, and be neither one nor every member"
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
            Self::SplitGathered { from, to } => write!(
                f,
                "Gathering a {from:?} tensor to {to:?} is refused: the sharded compute device \
                 was made with strict gathers"
            ),
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

#[cfg(test)]
mod tests {
    use alloc::borrow::ToOwned;

    use super::*;

    #[test]
    fn a_tally_since_an_earlier_one_holds_only_what_was_added() {
        let tally = |count, bytes| ReshardingTally {
            operation: "BaseFloat::Select".to_owned(),
            from: Sharding::Split { dim: 0 },
            to: Sharding::Replicated,
            count,
            bytes,
        };

        assert_eq!(tally(5, 500).since(&tally(2, 200)), tally(3, 300));
    }
}
