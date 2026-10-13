use std::ops::Range;

use burn_ir::TensorIr;
use burn_std::Shape;

/// How a tensor's shards, one per member of its group, make up its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupPlacement {
    /// Every member holds the whole value.
    Replicated,
    /// Balanced chunks in member order: the first `len % members` members hold one more element.
    Sharded { dim: usize },
    /// The global value is the sum of the shards.
    Partial,
}

/// How a placement follows its tensor's shape: the shard each member holds, and the placement
/// after a broadcast or a change in the number of dims.
pub trait PlacementShapes {
    /// An input of length 1 along the sharded dim broadcasts, so it stays whole.
    fn aligned(self, shape: &Shape) -> Self;

    /// The shape of the shard `member` holds of a tensor of global `shape`.
    fn local_shape(self, shape: &Shape, member: Member) -> Shape;

    /// The shard `member` holds of `tensor`, as its own interpreter knows it.
    fn local_tensor(self, tensor: &TensorIr, member: Member) -> TensorIr;

    /// The same placement once `count` leading dims are added, as broadcasting adds them.
    fn with_leading_dims(self, count: usize) -> Self;

    /// The same placement once the `count` leading dims are removed.
    ///
    /// # Panics
    ///
    /// When the sharded dim is one of them.
    fn without_leading_dims(self, count: usize) -> Self;
}

impl PlacementShapes for GroupPlacement {
    fn aligned(self, shape: &Shape) -> Self {
        match self {
            GroupPlacement::Sharded { dim } if shape[dim] == 1 => GroupPlacement::Replicated,
            placement => placement,
        }
    }

    fn local_shape(self, shape: &Shape, member: Member) -> Shape {
        let mut local = shape.clone();
        if let GroupPlacement::Sharded { dim } = self {
            let split = DimSplit {
                len: shape[dim],
                members: member.members,
            };
            local[dim] = split.range(member.index).len();
        }
        local
    }

    fn local_tensor(self, tensor: &TensorIr, member: Member) -> TensorIr {
        TensorIr {
            shape: self.local_shape(&tensor.shape, member),
            ..tensor.clone()
        }
    }

    fn with_leading_dims(self, count: usize) -> Self {
        match self {
            GroupPlacement::Sharded { dim } => GroupPlacement::Sharded { dim: dim + count },
            placement => placement,
        }
    }

    fn without_leading_dims(self, count: usize) -> Self {
        match self {
            GroupPlacement::Sharded { dim } => GroupPlacement::Sharded {
                dim: dim
                    .checked_sub(count)
                    .expect("The sharded dim is not a leading dim"),
            },
            placement => placement,
        }
    }
}

/// Where an op's inputs must be for it to run on every member at once, and where its output lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpPlacement<const N: usize> {
    pub inputs: [GroupPlacement; N],
    pub output: GroupPlacement,
}

/// One member of a group, by its place among all `members`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    pub index: usize,
    pub members: usize,
}

/// How a dim of `len` elements splits over `members`: the first `len % members` members hold
/// one more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimSplit {
    pub len: usize,
    pub members: usize,
}

impl DimSplit {
    pub fn range(&self, member: usize) -> Range<usize> {
        let (base, extra) = (self.len / self.members, self.len % self.members);
        let start = member * base + member.min(extra);
        start..start + base + usize::from(member < extra)
    }

    pub fn ranges(self) -> impl Iterator<Item = Range<usize>> {
        (0..self.members).map(move |member| self.range(member))
    }
}
