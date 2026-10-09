use burn_std::{DType, Shape};

use crate::{OpPlacement, Placement};

/// Which inputs of an elementwise op a partial sum can pass through: its output stays partial
/// only where the op distributes over the sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linearity {
    Nonlinear,
    /// Linear in every input at once: the output stays partial when every input is, like add.
    Linear,
    /// Linear in each input with the others fixed: one partial input among replicated ones
    /// keeps the output partial, like mul.
    Multilinear,
    /// Linear in one input with the others replicated: a dividend, or a gradient flowing back.
    LinearIn {
        input: usize,
    },
}

impl Linearity {
    /// Integer division rounds each summand, so only a float quotient stays partial.
    pub fn div(dtype: DType) -> Self {
        match dtype.is_float() {
            true => Linearity::LinearIn { input: 0 },
            false => Linearity::Nonlinear,
        }
    }

    /// Integer division rounds each summand, so only a float quotient stays partial.
    pub fn div_scalar(dtype: DType) -> Self {
        match dtype.is_float() {
            true => Linearity::Linear,
            false => Linearity::Nonlinear,
        }
    }

    /// Where the inputs must be for the op to run on every rank, and where its output lands.
    pub fn placement<const N: usize>(
        self,
        inputs: [Placement; N],
        shapes: [&Shape; N],
    ) -> OpPlacement<N> {
        if self.keeps_partial(&inputs) {
            return OpPlacement {
                inputs,
                output: Placement::Partial,
            };
        }

        let output = inputs
            .into_iter()
            .find(|placement| matches!(placement, Placement::Sharded { .. }))
            .unwrap_or(Placement::Replicated);
        OpPlacement {
            inputs: shapes.map(|shape| output.aligned(shape)),
            output,
        }
    }

    fn keeps_partial(self, inputs: &[Placement]) -> bool {
        let count = |placement| inputs.iter().filter(|input| **input == placement).count();
        let others_replicated = count(Placement::Replicated) == inputs.len() - 1;
        match self {
            Linearity::Nonlinear => false,
            Linearity::Linear => count(Placement::Partial) == inputs.len(),
            Linearity::Multilinear => count(Placement::Partial) == 1 && others_replicated,
            Linearity::LinearIn { input } => {
                inputs[input] == Placement::Partial && others_replicated
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Placement::{Partial, Replicated, Sharded};

    #[test]
    fn partial_plus_replicated_is_reduced_first() {
        let shape = Shape::new([4, 6]);
        let placement = Linearity::Linear.placement([Partial, Replicated], [&shape, &shape]);

        assert_eq!(placement.output, Replicated);
        assert_eq!(placement.inputs, [Replicated, Replicated]);
    }

    #[test]
    fn partial_times_replicated_stays_partial() {
        let shape = Shape::new([4, 6]);
        let placement = Linearity::Multilinear.placement([Replicated, Partial], [&shape, &shape]);

        assert_eq!(placement.output, Partial);
    }

    #[test]
    fn bias_broadcast_along_the_split_dim_stays_whole() {
        let placement = Linearity::Linear.placement(
            [Sharded { dim: 0 }, Replicated],
            [&Shape::new([4, 6]), &Shape::new([1, 6])],
        );

        assert_eq!(placement.inputs, [Sharded { dim: 0 }, Replicated]);
    }

    #[test]
    fn partial_meeting_a_split_input_is_reduce_scattered() {
        let shape = Shape::new([4, 6]);
        let placement =
            Linearity::Linear.placement([Partial, Sharded { dim: 1 }], [&shape, &shape]);

        assert_eq!(placement.inputs, [Sharded { dim: 1 }, Sharded { dim: 1 }]);
    }

    #[test]
    fn an_integer_quotient_of_a_partial_sum_is_reduced_first() {
        let shape = Shape::new([4]);
        let placement =
            Linearity::div(DType::I32).placement([Partial, Replicated], [&shape, &shape]);

        assert_eq!(placement.output, Replicated);
    }
}
