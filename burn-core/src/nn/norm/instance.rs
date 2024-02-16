use crate as burn;
use std::convert::Into;

use crate::config::Config;
use crate::module::Module;
use crate::tensor::{backend::Backend, Tensor};

use super::{GroupNorm, GroupNormConfig};

/// Configuration to create a [InstanceNorm](InstanceNorm) layer.
#[derive(Config)]
pub struct InstanceNormConfig {
    /// The number of channels expected in the input
    num_channels: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    epsilon: f64,
    /// A boolean value that when set to `true`, this module has learnable
    /// per-channel affine parameters initialized to ones (for weights)
    /// and zeros (for biases). Default: `true`
    #[config(default = true)]
    affine: bool,
}

/// Applies Instance Normalization over  a tensor as described in the paper [Instance Normalization](https://arxiv.org/abs/1607.08022)
#[derive(Module, Debug)]
pub struct InstanceNorm<B: Backend> {
    group_norm: GroupNorm<B>,
}

impl Into<GroupNormConfig> for &InstanceNormConfig {
    fn into(self) -> GroupNormConfig {
        GroupNormConfig {
            num_groups: self.num_channels,
            num_channels: self.num_channels,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }
}

impl InstanceNormConfig {
    /// Initialize a new [instance norm](InstanceNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> InstanceNorm<B> {
        InstanceNorm {
            group_norm: Into::<GroupNormConfig>::into(self).init(device),
        }
    }

    /// Initialize a new [instance norm](InstanceNorm) module with a [record](InstanceNormRecord).
    pub fn init_with<B: Backend>(&self, record: InstanceNormRecord<B>) -> InstanceNorm<B> {
        InstanceNorm {
            group_norm: Into::<GroupNormConfig>::into(self).init_with(record.group_norm),
        }
    }
}

impl<B: Backend> InstanceNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.group_norm.forward(input)
    }
}
