use burn_std::Shape;

use crate::{OpSharding, Sharding, ShardingOps};

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
    /// Linear in one input with the others replicated, like a dividend.
    LinearIn {
        input: usize,
    },
}

impl Linearity {
    /// Where the inputs must be for the op to run on every member, and where its output lands.
    pub fn sharding<const N: usize>(
        self,
        inputs: [Sharding; N],
        shapes: [&Shape; N],
    ) -> OpSharding<N> {
        if self.keeps_partial_sum(&inputs) {
            return OpSharding {
                inputs,
                output: Sharding::PartialSum,
            };
        }

        let output = inputs
            .into_iter()
            .find(|sharding| matches!(sharding, Sharding::Split { .. }))
            .unwrap_or(Sharding::Replicated);
        OpSharding {
            inputs: shapes.map(|shape| output.for_input(shape)),
            output,
        }
    }

    fn keeps_partial_sum(self, inputs: &[Sharding]) -> bool {
        let count = |sharding| inputs.iter().filter(|input| **input == sharding).count();
        let others_replicated = count(Sharding::Replicated) == inputs.len() - 1;
        match self {
            Linearity::Nonlinear => false,
            Linearity::Linear => count(Sharding::PartialSum) == inputs.len(),
            Linearity::Multilinear => count(Sharding::PartialSum) == 1 && others_replicated,
            Linearity::LinearIn { input } => {
                inputs[input] == Sharding::PartialSum && others_replicated
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::{PartialSum, Replicated, Split};

    #[test]
    fn partial_plus_replicated_is_reduced_first() {
        let shape = Shape::new([4, 6]);
        let sharding = Linearity::Linear.sharding([PartialSum, Replicated], [&shape, &shape]);

        assert_eq!(sharding.output, Replicated);
        assert_eq!(sharding.inputs, [Replicated, Replicated]);
    }

    #[test]
    fn partial_times_replicated_stays_partial() {
        let shape = Shape::new([4, 6]);
        let sharding = Linearity::Multilinear.sharding([Replicated, PartialSum], [&shape, &shape]);

        assert_eq!(sharding.output, PartialSum);
    }

    #[test]
    fn bias_broadcast_along_the_split_dim_stays_whole() {
        let sharding = Linearity::Linear.sharding(
            [Split { dim: 0 }, Replicated],
            [&Shape::new([4, 6]), &Shape::new([1, 6])],
        );

        assert_eq!(sharding.inputs, [Split { dim: 0 }, Replicated]);
    }

    #[test]
    fn partial_meeting_a_split_input_is_reduce_scattered() {
        let shape = Shape::new([4, 6]);
        let sharding = Linearity::Linear.sharding([PartialSum, Split { dim: 1 }], [&shape, &shape]);

        assert_eq!(sharding.inputs, [Split { dim: 1 }, Split { dim: 1 }]);
    }
}
