use burn_std::Shape;

use crate::Sharding::{self, Partial, Replicated, Split};

/// How the shards of a group's tensor move between members, from one sharding to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resharding {
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

impl Resharding {
    /// `None` when nothing can produce the target: a partial sum out of a tensor that is not one,
    /// or a split along a dim the tensor lacks or one shorter than the group, which would leave a
    /// member empty.
    pub fn new(from: Sharding, to: Sharding, shape: &Shape, members: usize) -> Option<Self> {
        if let Split { dim } = to
            && (dim >= shape.num_dims() || shape[dim] < members)
        {
            return None;
        }
        let resharding = match (from, to) {
            (Replicated, Replicated) | (Partial, Partial) => Self::Keep,
            (Split { dim: from }, Split { dim: to }) if from == to => Self::Keep,
            (Replicated, Split { dim }) => Self::Slice { dim },
            (Split { dim }, Replicated) => Self::AllGather { dim },
            (Split { dim: from_dim }, Split { dim: to_dim }) => Self::AllToAll { from_dim, to_dim },
            (Partial, Replicated) => Self::AllReduce,
            (Partial, Split { dim }) => Self::ReduceScatter { dim },
            (Replicated | Split { .. }, Partial) => return None,
        };
        Some(resharding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_along_a_dim_the_tensor_lacks_has_no_resharding() {
        let resharding = Resharding::new(Replicated, Split { dim: 2 }, &Shape::new([4, 6]), 2);

        assert_eq!(resharding, None);
    }

    #[test]
    fn only_a_partial_sum_stays_partial() {
        let shape = Shape::new([4, 6]);

        assert_eq!(Resharding::new(Split { dim: 0 }, Partial, &shape, 2), None);
        assert_eq!(
            Resharding::new(Partial, Partial, &shape, 2),
            Some(Resharding::Keep)
        );
    }
}
