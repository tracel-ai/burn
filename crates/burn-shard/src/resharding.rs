use burn_backend::ops::DimSplit;
use burn_ir::TensorIr;
use burn_std::{DType, Shape};

use crate::{
    Sharding::{self, PartialSum, Replicated, Split},
    ShardingError,
};

/// How the shards of a sharded compute device's tensor move between members, from one sharding to another.
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
    /// How `tensor` moves from `from` to `to`, or why it cannot: as [`new`](Self::new), and a
    /// quantized tensor's split must give each member whole quantization blocks and packed
    /// words, since its codes and scales are cut as they are.
    pub fn of_tensor(
        from: Sharding,
        to: Sharding,
        tensor: &TensorIr,
        num_members: usize,
    ) -> Result<Self, ShardingError> {
        let resharding = Self::new(from, to, &tensor.shape, num_members)?;
        let (DType::QFloat(scheme), Split { dim }) = (tensor.dtype, to) else {
            return Ok(resharding);
        };
        let shape = &tensor.shape;
        let split = DimSplit {
            dim,
            len: shape[dim],
            members: num_members,
        };
        match split.cuts_whole_quantized_units(&scheme, shape.num_dims()) {
            true => Ok(resharding),
            false => Err(ShardingError::QuantizedSplit {
                dim,
                shape: shape.clone(),
                num_members,
            }),
        }
    }

    /// How a value of `shape` moves from `from` to `to`, or why it cannot: a split needs the dim
    /// and an index per member along it.
    pub fn new(
        from: Sharding,
        to: Sharding,
        shape: &Shape,
        num_members: usize,
    ) -> Result<Self, ShardingError> {
        if let Split { dim } = to {
            if dim >= shape.num_dims() {
                return Err(ShardingError::MissingDim {
                    dim,
                    shape: shape.clone(),
                });
            }
            if shape[dim] < num_members {
                return Err(ShardingError::DimShorterThanMembers {
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
                return Err(ShardingError::PartialSumFromWhole { from });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use burn_ir::{TensorId, TensorStatus};
    use burn_std::{QuantScheme, ScaleDtype};

    use super::*;

    fn tensor(shape: impl Into<Shape>, dtype: DType) -> TensorIr {
        TensorIr {
            id: TensorId::new(0),
            shape: shape.into(),
            status: TensorStatus::ReadOnly,
            dtype,
        }
    }

    #[test]
    fn a_split_along_a_dim_the_tensor_lacks_has_no_resharding() {
        let resharding = Resharding::new(Replicated, Split { dim: 2 }, &Shape::new([4, 6]), 2);

        assert!(matches!(
            resharding,
            Err(ShardingError::MissingDim { dim: 2, .. })
        ));
    }

    #[test]
    fn a_dim_with_fewer_indices_than_members_has_no_resharding() {
        let resharding = Resharding::new(Replicated, Split { dim: 0 }, &Shape::new([2, 6]), 3);

        assert!(matches!(
            resharding,
            Err(ShardingError::DimShorterThanMembers { dim: 0, .. })
        ));
    }

    #[test]
    fn only_a_partial_sum_stays_partial() {
        let shape = Shape::new([4, 6]);

        assert_eq!(
            Resharding::new(Split { dim: 0 }, PartialSum, &shape, 2),
            Err(ShardingError::PartialSumFromWhole {
                from: Split { dim: 0 }
            })
        );
        assert_eq!(
            Resharding::new(PartialSum, PartialSum, &shape, 2),
            Ok(Resharding::Keep)
        );
    }

    #[test]
    fn a_quantized_split_that_cuts_a_block_has_no_resharding() {
        let scheme = QuantScheme::default().per_block([2], ScaleDtype::F32);
        let quantized = tensor([3, 6], DType::QFloat(scheme));

        assert!(matches!(
            Resharding::of_tensor(Replicated, Split { dim: 1 }, &quantized, 2),
            Err(ShardingError::QuantizedSplit { dim: 1, .. })
        ));
        assert_eq!(
            Resharding::of_tensor(Replicated, Split { dim: 0 }, &quantized, 2),
            Ok(Resharding::Slice { dim: 0 })
        );
    }
}
