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
    /// `None` when nothing can produce the target: a partial sum out of a tensor that is not one,
    /// or a split along a dim the tensor lacks or one shorter than the group, which would leave a
    /// member empty.
    pub fn new(
        from: GroupPlacement,
        to: GroupPlacement,
        shape: &Shape,
        members: usize,
    ) -> Option<Self> {
        if let Sharded { dim } = to
            && (dim >= shape.num_dims() || shape[dim] < members)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_along_a_dim_the_tensor_lacks_has_no_collective() {
        let redistribution =
            Redistribution::new(Replicated, Sharded { dim: 2 }, &Shape::new([4, 6]), 2);

        assert_eq!(redistribution, None);
    }

    #[test]
    fn only_a_partial_sum_stays_partial() {
        let shape = Shape::new([4, 6]);

        assert_eq!(
            Redistribution::new(Sharded { dim: 0 }, Partial, &shape, 2),
            None
        );
        assert_eq!(
            Redistribution::new(Partial, Partial, &shape, 2),
            Some(Redistribution::Keep)
        );
    }
}
