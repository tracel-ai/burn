use std::ops::Range;

use burn_ir::TensorIr;
use burn_std::Shape;

/// How a tensor's shards, one per member of its group, make up its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sharding {
    /// Every member holds the whole value.
    Replicated,
    /// Balanced chunks in member order: when the dim does not divide evenly, the first members hold
    /// one more element.
    Split { dim: usize },
    /// The global value is the sum of the shards.
    PartialSum,
}

/// How a sharding follows its tensor's shape: the shard each member holds, and the sharding
/// after a broadcast or a change in the number of dims.
pub trait ShardingOps {
    /// The sharding an input of `shape` takes beside an output of this sharding: an input of length
    /// 1 along the split dim broadcasts, so it stays whole.
    fn for_input(self, shape: &Shape) -> Self;

    /// The shape of the shard `member` holds of a tensor of global `shape`.
    fn local_shape(self, shape: &Shape, member: GroupMember) -> Shape;

    /// The shard `member` holds of `tensor`, as its own interpreter knows it.
    fn local_tensor(self, tensor: &TensorIr, member: GroupMember) -> TensorIr;

    /// The same sharding once `count` leading dims are added, as broadcasting adds them.
    fn with_leading_dims(self, count: usize) -> Self;

    /// The same sharding once the `count` leading dims are removed.
    ///
    /// # Panics
    ///
    /// When the split dim is one of them.
    fn without_leading_dims(self, count: usize) -> Self;
}

impl ShardingOps for Sharding {
    fn for_input(self, shape: &Shape) -> Self {
        match self {
            Sharding::Split { dim } if shape[dim] == 1 => Sharding::Replicated,
            sharding => sharding,
        }
    }

    fn local_shape(self, shape: &Shape, member: GroupMember) -> Shape {
        let mut local = shape.clone();
        if let Sharding::Split { dim } = self {
            let split = DimSplit {
                len: shape[dim],
                num_members: member.num_members,
            };
            local[dim] = split.range(member.index).len();
        }
        local
    }

    fn local_tensor(self, tensor: &TensorIr, member: GroupMember) -> TensorIr {
        TensorIr {
            shape: self.local_shape(&tensor.shape, member),
            ..tensor.clone()
        }
    }

    fn with_leading_dims(self, count: usize) -> Self {
        match self {
            Sharding::Split { dim } => Sharding::Split { dim: dim + count },
            sharding => sharding,
        }
    }

    fn without_leading_dims(self, count: usize) -> Self {
        match self {
            Sharding::Split { dim } => Sharding::Split {
                dim: dim
                    .checked_sub(count)
                    .expect("The split dim is not a leading dim"),
            },
            sharding => sharding,
        }
    }
}

/// Where an op's inputs must be for it to run on every member at once, and where its output lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpSharding<const N: usize> {
    pub inputs: [Sharding; N],
    pub output: Sharding,
}

/// One member of a group, by its index among the group's `num_members`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupMember {
    pub index: usize,
    pub num_members: usize,
}

/// How a dim of `len` elements splits over a group: the first `len % num_members` members hold one
/// more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimSplit {
    pub len: usize,
    pub num_members: usize,
}

impl DimSplit {
    pub fn range(&self, index: usize) -> Range<usize> {
        let (base, extra) = (self.len / self.num_members, self.len % self.num_members);
        let start = index * base + index.min(extra);
        start..start + base + usize::from(index < extra)
    }

    pub fn ranges(self) -> impl Iterator<Item = Range<usize>> {
        (0..self.num_members).map(move |index| self.range(index))
    }
}
