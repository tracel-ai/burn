use smallvec::smallvec;

use crate::{OpSharding, Sharding};

/// An op that treats each entry of its leading dims independently and reads the others whole,
/// like a pool over batch and channels: a split of any input along a leading dim splits them all
/// alike, and anything else gathers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialRule {
    sharding: Sharding,
}

impl SpatialRule {
    pub fn new(inputs: &[Sharding], leading: usize) -> Self {
        let sharding = inputs
            .iter()
            .copied()
            .find(|input| input.split_dim().is_some_and(|dim| dim < leading))
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
    use Sharding::{PartialSum, Replicated, ReplicatedSplit, Split};

    #[test]
    fn a_split_of_a_leading_dim_of_any_input_splits_them_all() {
        let channels = ReplicatedSplit { dim: 1, groups: 2 };
        let rule = SpatialRule::new(&[Replicated, channels], 2);

        assert_eq!(rule.sharding(2).inputs[..], [channels; 2]);
        assert_eq!(rule.sharding(2).output, channels);
    }

    #[test]
    fn a_split_of_a_dim_read_whole_or_a_partial_sum_gathers() {
        for sharding in [Split { dim: 1 }, Split { dim: 2 }, PartialSum] {
            assert_eq!(
                SpatialRule::new(&[sharding], 1).sharding(1).output,
                Replicated,
                "{sharding:?}"
            );
        }
    }
}
