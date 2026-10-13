use std::ops::Range;

use burn_ir::TensorIr;

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

impl GroupMember {
    /// The shard this member holds of `tensor`, as its own interpreter knows it.
    pub fn local_tensor(&self, tensor: &TensorIr, sharding: Sharding) -> TensorIr {
        let mut shape = tensor.shape.clone();
        if let Sharding::Split { dim } = sharding {
            let split = DimSplit {
                len: shape[dim],
                num_members: self.num_members,
            };
            shape[dim] = split.range(self.index).len();
        }
        TensorIr {
            shape,
            ..tensor.clone()
        }
    }
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
