use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Configuration to create a [InstanceNorm1d](InstanceNorm1d) layer.
#[derive(Config)]
pub struct InstanceNorm1dConfig {
    /// Number of features in the input tensor.
    pub num_features: usize,
    /// Epsilon value to avoid division by zero.
    pub eps: f64,
    /// Momentum value to update running statistics.
    pub momentum: f64,
    /// Whether to use the input tensor's mean and variance to normalize the tensor.
    pub affine: bool,
    /// Whether to use the input tensor's mean and variance to normalize the tensor.
    pub track_running_stats: bool,
}

/// Applies Instance Normalization over a mini-batch of inputs.
#[derive(Module, Debug)]
pub struct InstanceNorm1d<B: Backend> {
    num_features: usize,
    eps: f64,
    momentum: f64,
    affine: bool,
    track_running_stats: bool,
    weight: Option<Param<Tensor<B, 1>>>,
    bias: Option<Param<Tensor<B, 1>>>,
    running_mean: Option<Tensor<B, 1>>,
    running_var: Option<Tensor<B, 1>>,
}

impl InstanceNorm1dConfig {
    /// Initialize a new [instance norm](InstanceNorm1d) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> InstanceNorm1d<B> {
        let (weight, bias, running_mean, running_var) = if self.affine {
            let weight = Tensor::ones([self.num_features], device).into();
            let bias = Tensor::zeros([self.num_features], device).into();
            let running_mean = Tensor::zeros([self.num_features], device);
            let running_var = Tensor::ones([self.num_features], device);

            (
                Some(weight),
                Some(bias),
                Some(running_mean),
                Some(running_var),
            )
        } else {
            (None, None, None, None)
        };

        InstanceNorm1d {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            affine: self.affine,
            track_running_stats: self.track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
        }
    }
}

impl<B: Backend> InstanceNorm1d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// `Y = (X - μ) / σ * γ + β`
    pub fn forward(&self, input: &Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = input.mean_axis(&[2], false);
        let variance = input.var_axis(&[2], false);

        let mut output = input - mean;
        output /= variance.sqrt();
        output *= self.weight.as_ref().unwrap();
        output += self.bias.as_ref().unwrap();

        output
    }
}
