use burn_std::Shape;
use smallvec::SmallVec;

use crate::{Linearity, OpSharding, Sharding};

/// An elementwise op keeps a partial sum where its linearity lets one through, and otherwise
/// takes the first split among its inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementwiseRule {
    linearity: Linearity,
}

impl ElementwiseRule {
    pub fn new(linearity: Linearity) -> Self {
        Self { linearity }
    }

    /// Where the inputs must be for the op to run on every member, and where its output lands.
    pub fn sharding(self, inputs: &[Sharding], shapes: &[&Shape]) -> OpSharding {
        if self.keeps_partial_sum(inputs) {
            return OpSharding {
                inputs: SmallVec::from_slice(inputs),
                output: Sharding::PartialSum,
            };
        }

        let output = inputs
            .iter()
            .copied()
            .find(|sharding| matches!(sharding, Sharding::Split { .. }))
            .unwrap_or(Sharding::Replicated);
        OpSharding {
            inputs: shapes
                .iter()
                .map(|shape| Self::input_sharding(output, shape))
                .collect(),
            output,
        }
    }

    /// An input of length 1 along the split dim broadcasts, so it stays whole.
    pub fn input_sharding(output: Sharding, shape: &Shape) -> Sharding {
        match output {
            Sharding::Split { dim } if shape[dim] == 1 => Sharding::Replicated,
            sharding => sharding,
        }
    }

    fn keeps_partial_sum(self, inputs: &[Sharding]) -> bool {
        let count = |sharding| inputs.iter().filter(|input| **input == sharding).count();
        let others_replicated = count(Sharding::Replicated) == inputs.len() - 1;
        match self.linearity {
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
        let sharding = ElementwiseRule::new(Linearity::Linear)
            .sharding(&[PartialSum, Replicated], &[&shape, &shape]);

        assert_eq!(sharding.output, Replicated);
        assert_eq!(sharding.inputs[..], [Replicated, Replicated]);
    }

    #[test]
    fn partial_times_replicated_stays_partial() {
        let shape = Shape::new([4, 6]);
        let sharding = ElementwiseRule::new(Linearity::Multilinear)
            .sharding(&[Replicated, PartialSum], &[&shape, &shape]);

        assert_eq!(sharding.output, PartialSum);
    }

    #[test]
    fn bias_broadcast_along_the_split_dim_stays_whole() {
        let sharding = ElementwiseRule::new(Linearity::Linear).sharding(
            &[Split { dim: 0 }, Replicated],
            &[&Shape::new([4, 6]), &Shape::new([1, 6])],
        );

        assert_eq!(sharding.inputs[..], [Split { dim: 0 }, Replicated]);
    }

    #[test]
    fn partial_meeting_a_split_input_is_reduce_scattered() {
        let shape = Shape::new([4, 6]);
        let sharding = ElementwiseRule::new(Linearity::Linear)
            .sharding(&[PartialSum, Split { dim: 1 }], &[&shape, &shape]);

        assert_eq!(sharding.inputs[..], [Split { dim: 1 }, Split { dim: 1 }]);
    }
}
