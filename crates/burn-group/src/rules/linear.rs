use crate::{
    GroupPlacement::{self, Partial, Replicated, Sharded},
    MatmulRule, OpPlacement, PlacementShapes,
};

/// `x @ weight + bias`, with `weight` of shape `[d_in, d_out]`: the matmul of `x` with the
/// weight broadcast to the dims of `x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearRule {
    matmul: MatmulRule,
    num_dims: usize,
}

impl LinearRule {
    const WEIGHT_DIMS: usize = 2;
    /// A bias split with the output's features.
    const BIAS_SPLIT: GroupPlacement = Sharded { dim: 0 };

    /// The placements of `x`, of `num_dims` dims, and of `weight`.
    pub fn new(x: GroupPlacement, weight: GroupPlacement, num_dims: usize) -> Self {
        let leading_dims = num_dims - Self::WEIGHT_DIMS;
        let matmul = MatmulRule::new(x, weight.with_leading_dims(leading_dims), num_dims);
        Self { matmul, num_dims }
    }

    /// Inputs: x, weight, bias.
    pub fn placement(&self) -> OpPlacement<3> {
        let OpPlacement {
            inputs: [x, weight],
            output,
        } = self.matmul.placement();
        let bias = match output {
            Sharded { dim } if dim == self.num_dims - 1 => Self::BIAS_SPLIT,
            _ => Replicated,
        };
        OpPlacement {
            inputs: [
                x,
                weight.without_leading_dims(self.num_dims - Self::WEIGHT_DIMS),
                bias,
            ],
            output,
        }
    }

    /// A partial output takes the bias on one member only, or the sum would count it once per
    /// member.
    pub fn bias_on_one_member(&self) -> bool {
        self.placement().output == Partial
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_parallel_linear_splits_its_output_and_bias() {
        let rule = LinearRule::new(Replicated, Sharded { dim: 1 }, 3);

        assert_eq!(
            rule.placement(),
            OpPlacement {
                inputs: [Replicated, Sharded { dim: 1 }, Sharded { dim: 0 }],
                output: Sharded { dim: 2 },
            }
        );
    }

    #[test]
    fn row_parallel_linear_adds_its_bias_on_one_member() {
        let rule = LinearRule::new(Sharded { dim: 2 }, Sharded { dim: 0 }, 3);

        assert_eq!(rule.placement().output, Partial);
        assert!(rule.bias_on_one_member());
    }
}
