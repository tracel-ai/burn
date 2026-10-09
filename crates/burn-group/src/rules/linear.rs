use burn_std::Shape;

use crate::{
    GroupPlacement::{self, Partial, Replicated, Sharded},
    MatmulRule, OpPlacement,
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
    /// The weight's `d_in` dim, and the weight gradient matmul's lhs dim that holds it.
    const IN_FEATURES: usize = 0;
    /// The weight's `d_out` dim, and the weight gradient matmul's rhs dim that holds it.
    const OUT_FEATURES: usize = 1;
    const TRANSPOSE: [usize; 2] = [Self::OUT_FEATURES, Self::IN_FEATURES];
    /// A bias split with the output's features.
    const BIAS_SPLIT: GroupPlacement = Sharded { dim: 0 };

    /// The placements of `x` and of `weight`, with their shapes.
    pub fn new(
        x: GroupPlacement,
        weight: GroupPlacement,
        x_shape: &Shape,
        weight_shape: &Shape,
    ) -> Self {
        let num_dims = x_shape.num_dims();
        let matmul = MatmulRule::new(
            x,
            weight.with_num_dims(Self::WEIGHT_DIMS, num_dims),
            x_shape,
            &Self::broadcast(weight_shape, num_dims),
        );
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
                weight.with_num_dims(self.num_dims, Self::WEIGHT_DIMS),
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

    /// The gradient of `x`: `output_grad @ weight^T`. Inputs: weight, output gradient.
    pub fn x_backward(
        weight: GroupPlacement,
        output_grad: GroupPlacement,
        weight_shape: &Shape,
        grad_shape: &Shape,
    ) -> OpPlacement<2> {
        let num_dims = grad_shape.num_dims();
        let transposed = Shape::new(Self::TRANSPOSE.map(|dim| weight_shape[dim]));
        let matmul = MatmulRule::new(
            output_grad,
            weight
                .permuted(&Self::TRANSPOSE)
                .with_num_dims(Self::WEIGHT_DIMS, num_dims),
            grad_shape,
            &Self::broadcast(&transposed, num_dims),
        );
        let OpPlacement {
            inputs: [grad, weight],
            output,
        } = matmul.placement();
        OpPlacement {
            inputs: [
                weight
                    .with_num_dims(num_dims, Self::WEIGHT_DIMS)
                    .permuted(&Self::TRANSPOSE),
                grad,
            ],
            output,
        }
    }

    /// The gradient of the weight: `x^T @ output_grad`, summed over every batch dim, as one
    /// matmul whose contracted dim is the batch dims flattened. Inputs: x, output gradient.
    pub fn weight_backward(
        x: GroupPlacement,
        output_grad: GroupPlacement,
        x_shape: &Shape,
        grad_shape: &Shape,
    ) -> OpPlacement<2> {
        let num_dims = x_shape.num_dims();
        let features = num_dims - 1;
        let batch_dim = [x, output_grad]
            .into_iter()
            .find_map(|placement| match placement {
                Sharded { dim } if dim < features => Some(dim),
                _ => None,
            });
        // x^T is `[d_in, batch]` and the gradient `[batch, d_out]`, so each holds the batch in
        // the dim its features do not take.
        let flattened = |placement: GroupPlacement, feature_dim: usize| match placement {
            Sharded { dim } if dim == features => Sharded { dim: feature_dim },
            Sharded { dim } if Some(dim) == batch_dim => Sharded {
                dim: Self::OUT_FEATURES - feature_dim,
            },
            Sharded { .. } => Replicated,
            placement => placement,
        };
        let batch = x_shape[..features].iter().product::<usize>();
        let matmul = MatmulRule::new(
            flattened(x, Self::IN_FEATURES),
            flattened(output_grad, Self::OUT_FEATURES),
            &Shape::new([x_shape[features], batch]),
            &Shape::new([batch, grad_shape[features]]),
        );
        let OpPlacement {
            inputs: [lhs, rhs],
            output,
        } = matmul.placement();
        let unflattened = |placement: GroupPlacement, feature_dim: usize| match placement {
            Sharded { dim } if dim == feature_dim => Sharded { dim: features },
            Sharded { .. } => Sharded {
                dim: batch_dim.expect("A split contraction comes from a split batch dim"),
            },
            placement => placement,
        };
        OpPlacement {
            inputs: [
                unflattened(lhs, Self::IN_FEATURES),
                unflattened(rhs, Self::OUT_FEATURES),
            ],
            output,
        }
    }

    /// The gradient of the bias: the output gradient summed over every dim but the last.
    pub fn bias_backward(output_grad: GroupPlacement, grad_shape: &Shape) -> OpPlacement<1> {
        let features = grad_shape.num_dims() - 1;
        let output = match output_grad {
            Sharded { dim } if dim == features => Self::BIAS_SPLIT,
            Sharded { .. } | Partial => Partial,
            Replicated => Replicated,
        };
        OpPlacement {
            inputs: [output_grad],
            output,
        }
    }

    /// `shape` with leading dims of length 1 up to `num_dims`, as broadcasting sees it.
    fn broadcast(shape: &Shape, num_dims: usize) -> Shape {
        let mut dims = vec![1; num_dims - shape.num_dims()];
        dims.extend(shape.iter());
        Shape::from(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_parallel_linear_splits_its_output_and_bias() {
        let rule = LinearRule::new(
            Replicated,
            Sharded { dim: 1 },
            &Shape::new([2, 5, 6]),
            &Shape::new([6, 8]),
        );

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
        let rule = LinearRule::new(
            Sharded { dim: 2 },
            Sharded { dim: 0 },
            &Shape::new([2, 5, 8]),
            &Shape::new([8, 6]),
        );

        assert_eq!(rule.placement().output, Partial);
        assert!(rule.bias_on_one_member());
    }

    #[test]
    fn column_parallel_weight_gradient_stays_split() {
        let placement = LinearRule::weight_backward(
            Replicated,
            Sharded { dim: 2 },
            &Shape::new([2, 5, 6]),
            &Shape::new([2, 5, 8]),
        );

        assert_eq!(placement.output, Sharded { dim: 1 });
        assert_eq!(placement.inputs, [Replicated, Sharded { dim: 2 }]);
    }

    #[test]
    fn data_parallel_weight_gradient_is_a_partial_sum() {
        let placement = LinearRule::weight_backward(
            Sharded { dim: 0 },
            Sharded { dim: 0 },
            &Shape::new([4, 5, 6]),
            &Shape::new([4, 5, 8]),
        );

        assert_eq!(placement.output, Partial);
    }

    #[test]
    fn input_gradient_of_a_column_parallel_linear_is_a_partial_sum() {
        let placement = LinearRule::x_backward(
            Sharded { dim: 1 },
            Sharded { dim: 2 },
            &Shape::new([6, 8]),
            &Shape::new([2, 5, 8]),
        );

        assert_eq!(placement.output, Partial);
        assert_eq!(placement.inputs, [Sharded { dim: 1 }, Sharded { dim: 2 }]);
    }
}
