use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

#[derive(Config, Debug)]
pub struct InstanceNormConfig {
    /// The number of features.
    pub num_features: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// Momentum used to update the metrics. Default: 0.1
    #[config(default = 0.1)]
    pub momentum: f64,
    /// A boolean value that when set to `true`, this module has learnable
    /// affine parameters. Default: `false`
    #[config(default = false)]
    pub affine: bool,
}

#[derive(Module, Debug)]
pub struct InstanceNorm<B: Backend> {
    num_features: usize,
    epsilon: f64,
    momentum: f64,
    affine: bool,
    gamma: Option<Param<Tensor<B, 1>>>,
    beta: Option<Param<Tensor<B, 1>>>,
}

impl InstanceNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InstanceNorm<B> {
        let (gamma, beta) = if self.affine {
            let gamma = Tensor::ones([self.num_features], device).into();
            let beta = Tensor::zeros([self.num_features], device).into();
            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        InstanceNorm {
            num_features: self.num_features,
            epsilon: self.epsilon,
            momentum: self.momentum,
            affine: self.affine,
            gamma,
            beta,
        }
    }

    pub fn init_with<B: Backend>(&self, record: InstanceNormRecord<B>) -> InstanceNorm<B> {
        InstanceNorm {
            num_features: self.num_features,
            epsilon: self.epsilon,
            momentum: self.momentum,
            affine: self.affine,
            gamma: record.gamma,
            beta: record.beta,
        }
    }
}

impl<B: Backend> InstanceNorm<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mean = input.mean(&[-1], true);
        let variance = input.var(&[-1], true);
        let mut output = (input - mean) / variance.sqrt();
        if let Some(gamma) = &self.gamma {
            output *= gamma.value();
        }
        if let Some(beta) = &self.beta {
            output += beta.value();
        }
        output
    }
}
