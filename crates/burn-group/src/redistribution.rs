use burn_std::Shape;

use crate::GroupPlacement::{self, Partial, Replicated, Sharded};

/// The collective that moves a tensor from one placement to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Redistribution {
    Keep,
    /// Each member keeps its chunk of a replicated tensor.
    Slice {
        dim: usize,
    },
    AllGather {
        dim: usize,
    },
    AllReduce,
    /// Sum the summands, each member keeping its chunk.
    ReduceScatter {
        dim: usize,
    },
    AllToAll {
        from_dim: usize,
        to_dim: usize,
    },
}

impl Redistribution {
    /// `None` when nothing can produce the target: a partial sum out of a whole tensor, or a
    /// split along a dim shorter than the group, which would leave a member empty.
    pub fn new(
        from: GroupPlacement,
        to: GroupPlacement,
        shape: &Shape,
        members: usize,
    ) -> Option<Self> {
        if let Sharded { dim } = to
            && shape[dim] < members
        {
            return None;
        }
        let redistribution = match (from, to) {
            (Replicated, Replicated) | (Partial, Partial) => Self::Keep,
            (Sharded { dim: from }, Sharded { dim: to }) if from == to => Self::Keep,
            (Replicated, Sharded { dim }) => Self::Slice { dim },
            (Sharded { dim }, Replicated) => Self::AllGather { dim },
            (Sharded { dim: from_dim }, Sharded { dim: to_dim }) => {
                Self::AllToAll { from_dim, to_dim }
            }
            (Partial, Replicated) => Self::AllReduce,
            (Partial, Sharded { dim }) => Self::ReduceScatter { dim },
            (Replicated | Sharded { .. }, Partial) => return None,
        };
        Some(redistribution)
    }
}
