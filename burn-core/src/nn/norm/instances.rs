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
        let shape = input.shape();
        let rank = shape.num_elements();
        if D < 1 {
            panic!("D of InstanceNorm should be at least 1, but got {}", rank);
        }
        if !(rank == D + 1 || rank == D + 2) {
            panic!(
                "input rank for InstanceNorm should be either D+1 or D+2, but got {}",
                rank
            );
        }

        let batch_size = shape.dims[0];
        let num_channels = shape.dims[1];

        let hidden_size = shape.dims[2..].iter().product::<usize>();
        let input = input.reshape([batch_size, num_channels, hidden_size]);

        let mean = input.clone().sum_dim(2) / hidden_size as f64;
        let input = input.sub(mean);

        let var = input.clone().powf_scalar(2.).sum_dim(2) / hidden_size as f64;
        let input_normalized = input.div(var.sqrt().add_scalar(self.epsilon));

        if self.affine {
            let mut affine_shape = [1; D];
            affine_shape[1] = num_channels;

            input_normalized
                .reshape(shape)
                .mul(self.gamma.clone().unwrap().val().reshape(affine_shape))
                .add(self.beta.clone().unwrap().val().reshape(affine_shape))
        } else {
            input_normalized.reshape(shape)
        }
    }
}
