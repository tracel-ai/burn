use burn_backend::ops::DimSplit;
use burn_ir::TensorIr;
use burn_std::{Shape, distributed::Sharding};
use smallvec::SmallVec;

/// Attention's query, key, value, mask and bias are the most inputs a built-in op with a rule
/// reads, so planning one never allocates.
pub type OpInputs<T> = SmallVec<[T; 5]>;

/// A rule plans an op of one output.
pub type OpOutputs<T> = SmallVec<[T; 1]>;

/// Where an op's inputs must be for it to run on every member at once, and where its output lands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpSharding {
    /// One per input of the op, in its order.
    pub inputs: OpInputs<Sharding>,
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
        TensorIr {
            shape: self.local_shape(&tensor.shape, sharding),
            ..tensor.clone()
        }
    }

    /// The shape of the shard this member holds of a tensor of `shape`.
    pub fn local_shape(&self, shape: &Shape, sharding: Sharding) -> Shape {
        let mut shape = shape.clone();
        if let Sharding::Split { dim } = sharding {
            let split = DimSplit {
                dim,
                len: shape[dim],
                members: self.num_members,
            };
            shape[dim] = split.range(self.index).len();
        }
        shape
    }
}
