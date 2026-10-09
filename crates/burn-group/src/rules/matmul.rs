use burn_std::Shape;

use crate::{
    GroupPlacement::{self, Partial, Replicated, Sharded},
    OpPlacement,
};

/// Which dim of a matmul is split, named after the dim of the product it splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulRule {
    /// Nothing is split: every rank computes the whole product.
    Unsplit,
    /// The rhs columns are split, as in Megatron's column-parallel linear.
    Columns { dim: usize },
    /// The lhs rows are split.
    Rows { dim: usize },
    /// The contracted dim is split, as in Megatron's row-parallel linear: each rank computes
    /// a summand of the product.
    Contraction { lhs_dim: usize, rhs_dim: usize },
    /// A batch dim is split; an operand that broadcasts along it stays whole.
    Batches {
        dim: usize,
        lhs: GroupPlacement,
        rhs: GroupPlacement,
    },
    /// A partial lhs times a replicated rhs is the sum of each summand's product.
    PartialLhs,
    /// A replicated lhs times a partial rhs, likewise.
    PartialRhs,
    /// No split fits: both operands are gathered.
    Gathered,
}

impl MatmulRule {
    /// The placements of both operands, with their shapes.
    pub fn new(
        lhs: GroupPlacement,
        rhs: GroupPlacement,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Self {
        let num_dims = lhs_shape.num_dims();
        let (rows, cols) = (num_dims - 2, num_dims - 1);
        match (lhs, rhs) {
            (Replicated, Replicated) => Self::Unsplit,
            (Replicated, Sharded { dim }) if dim == cols => Self::Columns { dim },
            (Sharded { dim }, Replicated) if dim == rows => Self::Rows { dim },
            (Sharded { dim: l }, Sharded { dim: r }) if l == cols && r == rows => {
                Self::Contraction {
                    lhs_dim: cols,
                    rhs_dim: rows,
                }
            }
            (Sharded { dim }, Replicated) if dim == cols => Self::Contraction {
                lhs_dim: cols,
                rhs_dim: rows,
            },
            (Replicated, Sharded { dim }) if dim == rows => Self::Contraction {
                lhs_dim: cols,
                rhs_dim: rows,
            },
            (Sharded { dim: l }, Sharded { dim: r }) if l == r && l < rows => {
                Self::batches(l, lhs_shape, rhs_shape)
            }
            (Sharded { dim }, Replicated) | (Replicated, Sharded { dim }) if dim < rows => {
                Self::batches(dim, lhs_shape, rhs_shape)
            }
            (Partial, Replicated) => Self::PartialLhs,
            (Replicated, Partial) => Self::PartialRhs,
            // A partial operand is reduced to whatever the other one's split needs.
            (Partial, rhs) => Self::new(Replicated, rhs, lhs_shape, rhs_shape),
            (lhs, Partial) => Self::new(lhs, Replicated, lhs_shape, rhs_shape),
            _ => Self::Gathered,
        }
    }

    pub fn placement(&self) -> OpPlacement<2> {
        let (inputs, output) = match *self {
            Self::Unsplit | Self::Gathered => ([Replicated, Replicated], Replicated),
            Self::Columns { dim } => ([Replicated, Sharded { dim }], Sharded { dim }),
            Self::Rows { dim } => ([Sharded { dim }, Replicated], Sharded { dim }),
            Self::Contraction { lhs_dim, rhs_dim } => (
                [Sharded { dim: lhs_dim }, Sharded { dim: rhs_dim }],
                Partial,
            ),
            Self::Batches { dim, lhs, rhs } => ([lhs, rhs], Sharded { dim }),
            Self::PartialLhs => ([Partial, Replicated], Partial),
            Self::PartialRhs => ([Replicated, Partial], Partial),
        };
        OpPlacement { inputs, output }
    }

    fn batches(dim: usize, lhs: &Shape, rhs: &Shape) -> Self {
        let split = Sharded { dim };
        Self::Batches {
            dim,
            lhs: split.aligned(lhs),
            rhs: split.aligned(rhs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn megatron_mlp_runs_without_a_collective_between_its_matmuls() {
        let (x, w1, w2) = (Shape::new([4, 6]), Shape::new([6, 8]), Shape::new([8, 4]));
        let up = MatmulRule::new(Replicated, Sharded { dim: 1 }, &x, &w1);
        let down = MatmulRule::new(
            Sharded { dim: 1 },
            Sharded { dim: 0 },
            &Shape::new([4, 8]),
            &w2,
        );

        assert_eq!(up, MatmulRule::Columns { dim: 1 });
        assert_eq!(
            down.placement(),
            OpPlacement {
                inputs: [Sharded { dim: 1 }, Sharded { dim: 0 }],
                output: Partial,
            }
        );
    }

    #[test]
    fn partial_operand_is_reduce_scattered_into_a_contraction() {
        let rule = MatmulRule::new(
            Partial,
            Sharded { dim: 0 },
            &Shape::new([4, 8]),
            &Shape::new([8, 4]),
        );

        assert_eq!(
            rule,
            MatmulRule::Contraction {
                lhs_dim: 1,
                rhs_dim: 0
            }
        );
        assert_eq!(rule.placement().inputs[0], Sharded { dim: 1 });
    }

    #[test]
    fn operand_broadcast_along_the_split_batch_stays_whole() {
        let rule = MatmulRule::new(
            Sharded { dim: 0 },
            Replicated,
            &Shape::new([4, 2, 3]),
            &Shape::new([1, 3, 5]),
        );

        assert_eq!(rule.placement().inputs, [Sharded { dim: 0 }, Replicated]);
        assert_eq!(rule.placement().output, Sharded { dim: 0 });
    }

    #[test]
    fn rows_against_columns_falls_back_to_gathering() {
        let shape = Shape::new([4, 4]);
        let rule = MatmulRule::new(Sharded { dim: 0 }, Sharded { dim: 1 }, &shape, &shape);

        assert_eq!(rule, MatmulRule::Gathered);
    }
}
