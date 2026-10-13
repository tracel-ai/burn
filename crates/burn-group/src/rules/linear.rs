use crate::{
    MatmulRule, OpSharding, ShardShapes,
    Sharding::{self, Partial, Replicated, Split},
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
    const BIAS_SPLIT: Sharding = Split { dim: 0 };

    /// The shardings of `x`, of `num_dims` dims, and of `weight`.
    pub fn new(x: Sharding, weight: Sharding, num_dims: usize) -> Self {
        let leading_dims = num_dims - Self::WEIGHT_DIMS;
        let matmul = MatmulRule::new(x, weight.with_leading_dims(leading_dims), num_dims);
        Self { matmul, num_dims }
    }

    /// Inputs: x, weight, bias.
    pub fn sharding(&self) -> OpSharding<3> {
        let OpSharding {
            inputs: [x, weight],
            output,
        } = self.matmul.sharding();
        let bias = match output {
            Split { dim } if dim == self.num_dims - 1 => Self::BIAS_SPLIT,
            _ => Replicated,
        };
        OpSharding {
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
        self.sharding().output == Partial
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_parallel_linear_splits_its_output_and_bias() {
        let rule = LinearRule::new(Replicated, Split { dim: 1 }, 3);

        assert_eq!(
            rule.sharding(),
            OpSharding {
                inputs: [Replicated, Split { dim: 1 }, Split { dim: 0 }],
                output: Split { dim: 2 },
            }
        );
    }

    #[test]
    fn row_parallel_linear_adds_its_bias_on_one_member() {
        let rule = LinearRule::new(Split { dim: 2 }, Split { dim: 0 }, 3);

        assert_eq!(rule.sharding().output, Partial);
        assert!(rule.bias_on_one_member());
    }
}
