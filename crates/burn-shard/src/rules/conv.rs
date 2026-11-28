use crate::{
    ConvPass, OpSharding,
    Sharding::{self, PartialSum, Replicated, Split},
};

/// A convolution, or one of its gradients, split along the batch, its output channels or its
/// input channels, whichever its weight, bias, input or output's gradient is split along first.
/// A pass that sums over the split dim gives a partial sum, so its split must cut each entry to one
/// member. Grouped channels split only along the batch, since a member's channels would cut the
/// groups apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvRule {
    pass: ConvPass,
    split: Option<(ConvDim, Sharding)>,
    /// The weight's output then input channel dims.
    weight_channels: [usize; 2],
}

impl ConvRule {
    pub fn new(pass: ConvPass, inputs: &[Sharding], groups: usize, transposed: bool) -> Self {
        let weight_channels = if transposed { [1, 0] } else { [0, 1] };
        let tensors = &ConvTensor::inputs_of(pass)[..inputs.len()];
        let mut by_priority: Vec<_> = tensors
            .iter()
            .copied()
            .zip(inputs.iter().copied())
            .collect();
        by_priority.sort_by_key(|(tensor, _)| *tensor);
        let split = by_priority.into_iter().find_map(|(tensor, sharding)| {
            let dim = tensor.dim(sharding.split_dim()?, weight_channels)?;
            let fits = (groups == 1 || dim == ConvDim::Batch)
                && !(dim == ConvDim::InputChannels && tensors.contains(&ConvTensor::Bias))
                && (ConvDim::summed_by(pass) != dim || matches!(sharding, Split { .. }));
            fits.then_some((dim, sharding))
        });
        Self {
            pass,
            split,
            weight_channels,
        }
    }

    pub fn sharding(&self, num_inputs: usize) -> OpSharding {
        let of = |tensor: ConvTensor| match self.split {
            Some((dim, sharding)) => tensor.sharding(dim, sharding, self.weight_channels),
            None => Replicated,
        };
        let output = match self.split {
            Some((dim, _)) if dim == ConvDim::summed_by(self.pass) => PartialSum,
            _ => of(ConvTensor::result_of(self.pass)),
        };
        OpSharding {
            inputs: ConvTensor::inputs_of(self.pass)[..num_inputs]
                .iter()
                .map(|tensor| of(*tensor))
                .collect(),
            output,
        }
    }
}

/// A dim a convolution can split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConvDim {
    Batch,
    OutputChannels,
    InputChannels,
}

impl ConvDim {
    /// The dim `pass` sums over.
    fn summed_by(pass: ConvPass) -> Self {
        match pass {
            ConvPass::Forward => Self::InputChannels,
            ConvPass::InputGrad => Self::OutputChannels,
            ConvPass::WeightGrad | ConvPass::BiasGrad => Self::Batch,
        }
    }
}

/// A tensor a convolution reads or writes, in the order a split of one wins over another's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ConvTensor {
    Weight,
    Bias,
    Input,
    Output,
}

impl ConvTensor {
    /// The inputs of `pass` in order, a forward pass's bias last and optional.
    fn inputs_of(pass: ConvPass) -> [Self; 3] {
        match pass {
            ConvPass::Forward => [Self::Input, Self::Weight, Self::Bias],
            ConvPass::InputGrad | ConvPass::WeightGrad => [Self::Input, Self::Weight, Self::Output],
            ConvPass::BiasGrad => [Self::Input, Self::Bias, Self::Output],
        }
    }

    /// The tensor the result of `pass` is laid out as.
    fn result_of(pass: ConvPass) -> Self {
        match pass {
            ConvPass::Forward => Self::Output,
            ConvPass::InputGrad => Self::Input,
            ConvPass::WeightGrad => Self::Weight,
            ConvPass::BiasGrad => Self::Bias,
        }
    }

    /// The convolution's dim that a split of `split` cuts, if any.
    fn dim(self, split: usize, [output, input]: [usize; 2]) -> Option<ConvDim> {
        match (self, split) {
            (Self::Input | Self::Output, 0) => Some(ConvDim::Batch),
            (Self::Input, 1) => Some(ConvDim::InputChannels),
            (Self::Output, 1) | (Self::Bias, 0) => Some(ConvDim::OutputChannels),
            (Self::Weight, dim) if dim == output => Some(ConvDim::OutputChannels),
            (Self::Weight, dim) if dim == input => Some(ConvDim::InputChannels),
            _ => None,
        }
    }

    /// Its sharding when the convolution splits `dim` by `sharding`.
    fn sharding(self, dim: ConvDim, sharding: Sharding, [output, input]: [usize; 2]) -> Sharding {
        match (self, dim) {
            (Self::Input | Self::Output, ConvDim::Batch)
            | (Self::Bias, ConvDim::OutputChannels) => sharding.on_dim(0),
            (Self::Input, ConvDim::InputChannels) | (Self::Output, ConvDim::OutputChannels) => {
                sharding.on_dim(1)
            }
            (Self::Weight, ConvDim::OutputChannels) => sharding.on_dim(output),
            (Self::Weight, ConvDim::InputChannels) => sharding.on_dim(input),
            _ => Replicated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::ReplicatedSplit;

    #[test]
    fn a_split_batch_leaves_the_weight_whole() {
        let sharding =
            ConvRule::new(ConvPass::Forward, &[Split { dim: 0 }, Replicated], 1, false).sharding(2);

        assert_eq!(sharding.inputs[..], [Split { dim: 0 }, Replicated]);
        assert_eq!(sharding.output, Split { dim: 0 });
    }

    #[test]
    fn a_weight_split_by_output_channels_splits_the_bias_and_the_output_channels() {
        let weight = Split { dim: 0 };
        let rule = ConvRule::new(
            ConvPass::Forward,
            &[Split { dim: 0 }, weight, Replicated],
            1,
            false,
        );

        assert_eq!(rule.sharding(3).inputs[..], [Replicated, weight, weight]);
        assert_eq!(rule.sharding(3).output, Split { dim: 1 });
    }

    #[test]
    fn a_transposed_weight_holds_its_output_channels_second() {
        let weight = Split { dim: 1 };
        let rule = ConvRule::new(ConvPass::Forward, &[Replicated, weight], 1, true);

        assert_eq!(rule.sharding(2).output, Split { dim: 1 });
        assert_eq!(
            ConvRule::new(ConvPass::Forward, &[Replicated, Split { dim: 0 }], 1, true)
                .sharding(2)
                .output,
            PartialSum
        );
    }

    #[test]
    fn split_input_channels_give_a_partial_sum_without_a_bias() {
        let input = Split { dim: 1 };
        let unbiased = ConvRule::new(ConvPass::Forward, &[input, Replicated], 1, false);
        let biased = ConvRule::new(
            ConvPass::Forward,
            &[input, Replicated, Replicated],
            1,
            false,
        );

        assert_eq!(unbiased.sharding(2).inputs[..], [input; 2]);
        assert_eq!(unbiased.sharding(2).output, PartialSum);
        assert_eq!(biased.sharding(3).output, Replicated);
    }

    #[test]
    fn each_gradient_sums_over_the_dim_its_pass_contracts() {
        let batch = [Split { dim: 0 }, Replicated, Split { dim: 0 }];
        let output_channels = [Replicated, Split { dim: 0 }, Split { dim: 1 }];

        assert_eq!(
            ConvRule::new(ConvPass::InputGrad, &output_channels, 1, false)
                .sharding(3)
                .output,
            PartialSum
        );
        assert_eq!(
            ConvRule::new(ConvPass::WeightGrad, &batch, 1, false)
                .sharding(3)
                .output,
            PartialSum
        );
        assert_eq!(
            ConvRule::new(ConvPass::BiasGrad, &output_channels, 1, false)
                .sharding(3)
                .output,
            Split { dim: 0 }
        );
    }

    #[test]
    fn a_sum_over_a_split_in_groups_of_members_gathers() {
        let batch = ReplicatedSplit { dim: 0, groups: 2 };
        let rule = ConvRule::new(ConvPass::WeightGrad, &[batch, Replicated, batch], 1, false);

        assert_eq!(rule.sharding(3).output, Replicated);
    }

    #[test]
    fn grouped_channels_split_only_along_the_batch() {
        let rule = ConvRule::new(
            ConvPass::Forward,
            &[Split { dim: 1 }, Split { dim: 0 }],
            2,
            false,
        );

        assert_eq!(rule.sharding(2).output, Replicated);
    }
}
