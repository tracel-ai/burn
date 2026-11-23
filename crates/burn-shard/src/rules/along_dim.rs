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
        let sharding = match input.split_dim() {
            Some(split) if split != dim => input,
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
    use Sharding::{PartialSum, Replicated, ReplicatedSplit, Split};

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

    #[test]
    fn a_norm_over_each_head_keeps_heads_shared_by_pairs() {
        let kv = ReplicatedSplit { dim: 1, groups: 2 };

        assert_eq!(AlongDimRule::new(kv, 3).sharding().output, kv);
    }
}
