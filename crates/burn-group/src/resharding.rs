use core::fmt;

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

/// Why no resharding produces a target sharding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReshardingError {
    /// The split dim is not one of the tensor's.
    MissingDim { dim: usize, shape: Shape },
    /// Fewer indices than members along the split dim would leave a member empty.
    DimShorterThanGroup {
        dim: usize,
        shape: Shape,
        num_members: usize,
    },
    /// A partial sum is an op's output, never a split of a whole value.
    PartialSumFromWhole { from: Sharding },
}

impl Resharding {
    pub fn new(
        from: Sharding,
        to: Sharding,
        shape: &Shape,
        num_members: usize,
    ) -> Result<Self, ReshardingError> {
        if let Split { dim } = to {
            if dim >= shape.num_dims() {
                return Err(ReshardingError::MissingDim {
                    dim,
                    shape: shape.clone(),
                });
            }
            if shape[dim] < num_members {
                return Err(ReshardingError::DimShorterThanGroup {
                    dim,
                    shape: shape.clone(),
                    num_members,
                });
            }
        }
        Ok(match (from, to) {
            (Replicated, Replicated) | (PartialSum, PartialSum) => Self::Keep,
            (Split { dim: from }, Split { dim: to }) if from == to => Self::Keep,
            (Replicated, Split { dim }) => Self::Slice { dim },
            (Split { dim }, Replicated) => Self::AllGather { dim },
            (Split { dim: from_dim }, Split { dim: to_dim }) => Self::AllToAll { from_dim, to_dim },
            (PartialSum, Replicated) => Self::AllReduce,
            (PartialSum, Split { dim }) => Self::ReduceScatter { dim },
            (Replicated | Split { .. }, PartialSum) => {
                return Err(ReshardingError::PartialSumFromWhole { from });
            }
        })
    }
}

impl fmt::Display for ReshardingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDim { dim, shape } => {
                write!(f, "A tensor of shape {shape:?} has no dim {dim} to split")
            }
            Self::DimShorterThanGroup {
                dim,
                shape,
                num_members,
            } => write!(
                f,
                "Dim {dim} of a tensor of shape {shape:?} is shorter than the group's \
                 {num_members} members, so a split would leave a member empty"
            ),
            Self::PartialSumFromWhole { from } => {
                write!(
                    f,
                    "A {from:?} tensor is not a partial sum and cannot become one"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_along_a_dim_the_tensor_lacks_has_no_resharding() {
        let resharding = Resharding::new(Replicated, Split { dim: 2 }, &Shape::new([4, 6]), 2);

        assert!(matches!(
            resharding,
            Err(ReshardingError::MissingDim { dim: 2, .. })
        ));
    }

    #[test]
    fn a_dim_shorter_than_the_group_has_no_resharding() {
        let resharding = Resharding::new(Replicated, Split { dim: 0 }, &Shape::new([2, 6]), 3);

        assert!(matches!(
            resharding,
            Err(ReshardingError::DimShorterThanGroup { dim: 0, .. })
        ));
    }

    #[test]
    fn only_a_partial_sum_stays_partial() {
        let shape = Shape::new([4, 6]);

        assert_eq!(
            Resharding::new(Split { dim: 0 }, PartialSum, &shape, 2),
            Err(ReshardingError::PartialSumFromWhole {
                from: Split { dim: 0 }
            })
        );
        assert_eq!(
            Resharding::new(PartialSum, PartialSum, &shape, 2),
            Ok(Resharding::Keep)
        );
    }
}
