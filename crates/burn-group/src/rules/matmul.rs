use crate::{
    GroupPlacement::{self, Partial, Replicated, Sharded},
    OpPlacement,
};

/// Which dim of a matmul is split, named after the dim of the product it splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulRule {
    /// Nothing is split: every member computes the whole product.
    Unsplit,
    /// The rhs columns are split, as in Megatron's column-parallel linear.
    Columns { dim: usize },
    /// The contracted dim is split, as in Megatron's row-parallel linear: each member computes
    /// a summand of the product.
    Contraction { lhs_dim: usize, rhs_dim: usize },
    /// No split fits: both operands are gathered.
    Gathered,
}

impl MatmulRule {
    /// The placements of both operands, of `num_dims` dims each.
    pub fn new(lhs: GroupPlacement, rhs: GroupPlacement, num_dims: usize) -> Self {
        let (rows, cols) = (num_dims - 2, num_dims - 1);
        let contraction = Self::Contraction {
            lhs_dim: cols,
            rhs_dim: rows,
        };
        match (lhs, rhs) {
            (Replicated, Replicated) => Self::Unsplit,
            (Replicated, Sharded { dim }) if dim == cols => Self::Columns { dim },
            (Sharded { dim: l }, Sharded { dim: r }) if l == cols && r == rows => contraction,
            (Sharded { dim }, Replicated) if dim == cols => contraction,
            (Replicated, Sharded { dim }) if dim == rows => contraction,
            // A partial operand is reduced to whatever the other one's split needs.
            (Partial, rhs) => Self::new(Replicated, rhs, num_dims),
            (lhs, Partial) => Self::new(lhs, Replicated, num_dims),
            _ => Self::Gathered,
        }
    }

    pub fn placement(&self) -> OpPlacement<2> {
        let (inputs, output) = match *self {
            Self::Unsplit | Self::Gathered => ([Replicated, Replicated], Replicated),
            Self::Columns { dim } => ([Replicated, Sharded { dim }], Sharded { dim }),
            Self::Contraction { lhs_dim, rhs_dim } => (
                [Sharded { dim: lhs_dim }, Sharded { dim: rhs_dim }],
                Partial,
            ),
        };
        OpPlacement { inputs, output }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn megatron_mlp_runs_without_a_collective_between_its_matmuls() {
        let up = MatmulRule::new(Replicated, Sharded { dim: 1 }, 2);
        let down = MatmulRule::new(Sharded { dim: 1 }, Sharded { dim: 0 }, 2);

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
        let rule = MatmulRule::new(Partial, Sharded { dim: 0 }, 2);

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
    fn rows_against_columns_falls_back_to_gathering() {
        let rule = MatmulRule::new(Sharded { dim: 0 }, Sharded { dim: 1 }, 2);

        assert_eq!(rule, MatmulRule::Gathered);
    }
}
