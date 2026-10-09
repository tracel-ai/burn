use std::ops::Range;

use burn_std::Shape;

/// How a tensor's shards, one per rank of its group, make up its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Every rank holds the whole value.
    Replicated,
    /// Balanced chunks in rank order: the first `len % ranks` ranks hold one more element.
    Sharded { dim: usize },
    /// The global value is the sum of the shards.
    Partial,
}

impl Placement {
    /// An input of length 1 along the sharded dim broadcasts, so it stays whole.
    pub fn aligned(self, shape: &Shape) -> Self {
        match self {
            Placement::Sharded { dim } if shape[dim] == 1 => Placement::Replicated,
            placement => placement,
        }
    }

    /// The shape of the shard `rank` holds of a tensor of global `shape`.
    pub fn local_shape(self, shape: &Shape, rank: usize, ranks: usize) -> Shape {
        let mut local = shape.clone();
        if let Placement::Sharded { dim } = self {
            local[dim] = Chunks::new(shape[dim], ranks).range(rank).len();
        }
        local
    }

    /// The placement of a tensor's permutation by `axes`: the sharded dim follows its axis.
    pub fn permuted(self, axes: &[usize]) -> Self {
        match self {
            Placement::Sharded { dim } => Placement::Sharded {
                dim: axes
                    .iter()
                    .position(|axis| *axis == dim)
                    .expect("Axes must be a permutation"),
            },
            placement => placement,
        }
    }

    /// The same placement on a tensor of `new_num_dims` dims instead of `num_dims`, with
    /// trailing dims lined up as broadcasting does.
    pub fn with_num_dims(self, num_dims: usize, new_num_dims: usize) -> Self {
        match self {
            Placement::Sharded { dim } => Placement::Sharded {
                dim: (dim + new_num_dims)
                    .checked_sub(num_dims)
                    .expect("The sharded dim exists in the new dims"),
            },
            placement => placement,
        }
    }
}

/// Where an op's inputs must be for it to run on every rank at once, and where its output lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpPlacement<const N: usize> {
    pub inputs: [Placement; N],
    pub output: Placement,
}

/// How a dim of `len` elements splits over `ranks`: the first `len % ranks` ranks hold one more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunks {
    len: usize,
    ranks: usize,
}

impl Chunks {
    pub fn new(len: usize, ranks: usize) -> Self {
        Self { len, ranks }
    }

    pub fn range(&self, rank: usize) -> Range<usize> {
        let (base, extra) = (self.len / self.ranks, self.len % self.ranks);
        let start = rank * base + rank.min(extra);
        start..start + base + usize::from(rank < extra)
    }

    pub fn ranges(self) -> impl Iterator<Item = Range<usize>> {
        (0..self.ranks).map(move |rank| self.range(rank))
    }
}
