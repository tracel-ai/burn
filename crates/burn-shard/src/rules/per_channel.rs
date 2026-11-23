use std::iter;

use crate::{OpSharding, Sharding};

/// An op per element whose further inputs hold one entry per channel, like a batch norm: a split
/// of the channels splits those inputs alike, a split of another dim leaves them whole, and a
/// partial sum gathers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerChannelRule {
    input: Sharding,
    channels: Sharding,
}

impl PerChannelRule {
    pub fn new(input: Sharding) -> Self {
        match input.split_dim() {
            Some(1) => Self {
                input,
                channels: input.on_dim(0),
            },
            Some(_) => Self {
                input,
                channels: Sharding::Replicated,
            },
            None => Self {
                input: Sharding::Replicated,
                channels: Sharding::Replicated,
            },
        }
    }

    pub fn sharding(&self, num_per_channel: usize) -> OpSharding {
        let channels = iter::repeat_n(self.channels, num_per_channel);
        OpSharding {
            inputs: [self.input].into_iter().chain(channels).collect(),
            output: self.input,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::{PartialSum, Replicated, Split};

    #[test]
    fn a_split_of_the_channels_splits_each_channel_input() {
        let sharding = PerChannelRule::new(Split { dim: 1 }).sharding(2);

        assert_eq!(
            sharding.inputs[..],
            [Split { dim: 1 }, Split { dim: 0 }, Split { dim: 0 }]
        );
        assert_eq!(sharding.output, Split { dim: 1 });
    }

    #[test]
    fn a_split_of_another_dim_leaves_the_channel_inputs_whole() {
        let sharding = PerChannelRule::new(Split { dim: 2 }).sharding(1);

        assert_eq!(sharding.inputs[..], [Split { dim: 2 }, Replicated]);
    }

    #[test]
    fn a_partial_sum_gathers() {
        assert_eq!(
            PerChannelRule::new(PartialSum).sharding(1).output,
            Replicated
        );
    }
}
