use std::fmt;

use burn_backend::ops::DimSplit;
use burn_ir::TensorIr;
use burn_std::Shape;

/// How a tensor's shards, one per member of its sharded compute device, make up its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sharding {
    /// Every member holds the whole value.
    Replicated,
    /// Balanced chunks in member order: when the dim does not divide evenly, the first members each
    /// hold one more index along `dim`.
    Split { dim: usize },
    /// The global value is the sum of the shards.
    PartialSum,
}

/// Why a tensor cannot take a sharding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardingError {
    /// Only a tensor on a sharded compute device has a sharding.
    NotSharded,
    /// Quantized tensors are not split over a sharded compute device.
    Quantized,
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

impl fmt::Display for ShardingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSharded => write!(
                f,
                "Only a tensor on a sharded compute device has a sharding"
            ),
            Self::Quantized => write!(
                f,
                "Quantized tensors are not split over a sharded compute device"
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

impl std::error::Error for ShardingError {}

/// Where an op's inputs must be for it to run on every member at once, and where its output lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpSharding<const N: usize> {
    pub inputs: [Sharding; N],
    pub output: Sharding,
}

/// One member of a sharded compute device, by its index among the sharded compute device's `num_members`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardMember {
    pub index: usize,
    pub num_members: usize,
}

impl ShardMember {
    /// The shard this member holds of `tensor`, as its own interpreter knows it.
    pub fn local_tensor(&self, tensor: &TensorIr, sharding: Sharding) -> TensorIr {
        let mut shape = tensor.shape.clone();
        if let Sharding::Split { dim } = sharding {
            let split = DimSplit {
                dim,
                len: shape[dim],
                members: self.num_members,
            };
            shape[dim] = split.range(self.index).len();
        }
        TensorIr {
            shape,
            ..tensor.clone()
        }
    }
}
