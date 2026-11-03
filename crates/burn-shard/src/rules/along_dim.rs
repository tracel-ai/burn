use smallvec::smallvec;

use crate::{OpSharding, Sharding};

/// An op that reads one dim whole and treats every other independently, like softmax, cumsum or
/// sort: a split elsewhere stays, and a split of that dim or a partial sum gathers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlongDimRule {
    sharding: Sharding,
}

impl AlongDimRule {
    pub fn new(input: Sharding, dim: usize) -> Self {
        let sharding = match input {
            Sharding::Split { dim: split } if split != dim => input,
            _ => Sharding::Replicated,
        };
        Self { sharding }
    }

    pub fn sharding(&self) -> OpSharding {
        OpSharding {
            inputs: smallvec![self.sharding],
            output: self.sharding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::{PartialSum, Replicated, Split};

    #[test]
    fn a_split_of_another_dim_stays() {
        assert_eq!(
            AlongDimRule::new(Split { dim: 0 }, 1).sharding().output,
            Split { dim: 0 }
        );
    }

    #[test]
    fn a_split_of_the_dim_or_a_partial_sum_gathers() {
        assert_eq!(
            AlongDimRule::new(Split { dim: 1 }, 1).sharding().inputs[..],
            [Replicated]
        );
        assert_eq!(
            AlongDimRule::new(PartialSum, 1).sharding().inputs[..],
            [Replicated]
        );
    }
}
