use burn_std::Shape;
use smallvec::smallvec;

use crate::{OpSharding, Sharding};

/// An op that indexes along one dim and pairs its inputs entry by entry on every other, like
/// gather and scatter: a split of any input along another dim, which every input has at the same
/// length, splits them all alike. Burn lets an index tensor be shorter there, which gathers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexRule {
    sharding: Sharding,
}

impl IndexRule {
    pub fn new(inputs: &[Sharding], shapes: &[&Shape], dim: usize) -> Self {
        let paired = |split: usize| {
            let len = |shape: &Shape| (split < shape.num_dims()).then(|| shape[split]);
            shapes.iter().all(|shape| len(shape) == len(shapes[0]))
        };
        let sharding = inputs
            .iter()
            .copied()
            .find(|input| {
                input
                    .split_dim()
                    .is_some_and(|split| split != dim && paired(split))
            })
            .unwrap_or(Sharding::Replicated);
        Self { sharding }
    }

    pub fn sharding(&self, num_inputs: usize) -> OpSharding {
        OpSharding {
            inputs: smallvec![self.sharding; num_inputs],
            output: self.sharding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::{PartialSum, Replicated, Split};

    #[test]
    fn a_split_of_the_indices_along_another_dim_splits_the_input_alike() {
        let shape = Shape::new([4, 6]);
        let rule = IndexRule::new(&[Replicated, Split { dim: 0 }], &[&shape, &shape], 1);

        assert_eq!(rule.sharding(2).inputs[..], [Split { dim: 0 }; 2]);
        assert_eq!(rule.sharding(2).output, Split { dim: 0 });
    }

    #[test]
    fn a_split_of_the_indexed_dim_or_of_shorter_indices_gathers() {
        let (input, shorter) = (Shape::new([4, 6]), Shape::new([2, 6]));
        for (sharding, indices) in [(Split { dim: 1 }, &input), (Split { dim: 0 }, &shorter)] {
            let rule = IndexRule::new(&[sharding, Replicated], &[&input, indices], 1);

            assert_eq!(rule.sharding(2).output, Replicated, "{sharding:?}");
        }
    }

    #[test]
    fn a_partial_sum_gathers() {
        let shape = Shape::new([4, 6]);
        let rule = IndexRule::new(&[PartialSum, Replicated], &[&shape, &shape], 1);

        assert_eq!(rule.sharding(2).inputs[..], [Replicated; 2]);
    }
}
