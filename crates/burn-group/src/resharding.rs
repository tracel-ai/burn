use burn_std::Shape;

use crate::Sharding::{self, PartialSum, Replicated, Split};

/// How the shards of a group's tensor move between members, from one sharding to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resharding {
    /// The tensor already has the target sharding, so nothing moves.
    Keep,
    /// Replicated to split: each member keeps its own chunk along `dim`, with nothing sent.
    Slice { dim: usize },
    /// Split to replicated: every member receives every other member's chunk along `dim`.
    AllGather { dim: usize },
    /// Partial sum to replicated: every member ends with the sum of every member's shard.
    AllReduce,
    /// Partial sum to split: the shards are summed, each member keeping only its chunk along
    /// `dim` of the total.
    ReduceScatter { dim: usize },
    /// Split along `from_dim` to split along `to_dim`: each member sends every other member the
    /// part of its chunk that the other holds once split along `to_dim`.
    AllToAll { from_dim: usize, to_dim: usize },
}

impl Resharding {
    /// `None` when nothing can produce the target: a partial sum out of a tensor that is not one,
    /// or a split along a dim the tensor lacks or one shorter than the group, which would leave a
    /// member empty.
    pub fn new(from: Sharding, to: Sharding, shape: &Shape, num_members: usize) -> Option<Self> {
        if let Split { dim } = to
            && (dim >= shape.num_dims() || shape[dim] < num_members)
        {
            return None;
        }
        let resharding = match (from, to) {
            (Replicated, Replicated) | (PartialSum, PartialSum) => Self::Keep,
            (Split { dim: from }, Split { dim: to }) if from == to => Self::Keep,
            (Replicated, Split { dim }) => Self::Slice { dim },
            (Split { dim }, Replicated) => Self::AllGather { dim },
            (Split { dim: from_dim }, Split { dim: to_dim }) => Self::AllToAll { from_dim, to_dim },
            (PartialSum, Replicated) => Self::AllReduce,
            (PartialSum, Split { dim }) => Self::ReduceScatter { dim },
            (Replicated | Split { .. }, PartialSum) => return None,
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

        assert_eq!(
            Resharding::new(Split { dim: 0 }, PartialSum, &shape, 2),
            None
        );
        assert_eq!(
            Resharding::new(PartialSum, PartialSum, &shape, 2),
            Some(Resharding::Keep)
        );
    }
}
