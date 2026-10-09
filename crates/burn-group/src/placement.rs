use std::ops::Range;

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
/// after a broadcast, a permute or a change in the number of dims.
pub trait PlacementShapes {
    /// An input of length 1 along the sharded dim broadcasts, so it stays whole.
    fn aligned(self, shape: &Shape) -> Self;

    /// The shape of the shard `member` holds of a tensor of global `shape`.
    fn local_shape(self, shape: &Shape, member: usize, members: usize) -> Shape;

    /// The placement of a tensor's permutation by `axes`: the sharded dim follows its axis.
    fn permuted(self, axes: &[usize]) -> Self;

    /// The same placement on a tensor of `new_num_dims` dims instead of `num_dims`, with
    /// trailing dims lined up as broadcasting does.
    fn with_num_dims(self, num_dims: usize, new_num_dims: usize) -> Self;
}

impl PlacementShapes for GroupPlacement {
    fn aligned(self, shape: &Shape) -> Self {
        match self {
            GroupPlacement::Sharded { dim } if shape[dim] == 1 => GroupPlacement::Replicated,
            placement => placement,
        }
    }

    fn local_shape(self, shape: &Shape, member: usize, members: usize) -> Shape {
        let mut local = shape.clone();
        if let GroupPlacement::Sharded { dim } = self {
            local[dim] = DimSplit::new(shape[dim], members).range(member).len();
        }
        local
    }

    fn permuted(self, axes: &[usize]) -> Self {
        match self {
            GroupPlacement::Sharded { dim } => GroupPlacement::Sharded {
                dim: axes
                    .iter()
                    .position(|axis| *axis == dim)
                    .expect("Axes must be a permutation"),
            },
            placement => placement,
        }
    }

    fn with_num_dims(self, num_dims: usize, new_num_dims: usize) -> Self {
        match self {
            GroupPlacement::Sharded { dim } => GroupPlacement::Sharded {
                dim: (dim + new_num_dims)
                    .checked_sub(num_dims)
                    .expect("The sharded dim exists in the new dims"),
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

/// How a dim of `len` elements splits over `members`: the first `len % members` members hold
/// one more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimSplit {
    len: usize,
    members: usize,
}

impl DimSplit {
    pub fn new(len: usize, members: usize) -> Self {
        Self { len, members }
    }

    pub fn range(&self, member: usize) -> Range<usize> {
        let (base, extra) = (self.len / self.members, self.len % self.members);
        let start = member * base + member.min(extra);
        start..start + base + usize::from(member < extra)
    }

    pub fn ranges(self) -> impl Iterator<Item = Range<usize>> {
        (0..self.members).map(move |member| self.range(member))
    }
}
