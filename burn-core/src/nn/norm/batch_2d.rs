use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::module::RunningState;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Configuration to create a [BatchNorm2d](BatchNorm2d) layer.
#[derive(Config)]
pub struct BatchNorm2dConfig {
    /// The number of features.
    pub num_features: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// Momentum used to update the metrics. Default: 0.1
    #[config(default = 0.1)]
    pub momentum: f64,
}

/// Applies Batch Normalization over a 4D tensor as described in the paper [Batch Normalization](https://arxiv.org/abs/1502.03167)
///
/// `Y = norm(X) * γ + β`
#[derive(Module, Debug)]
pub struct BatchNorm2d<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    running_mean: Param<RunningState<Tensor<B, 1>>>,
    running_var: Param<RunningState<Tensor<B, 1>>>,
    epsilon: f64,
}

impl<B: Backend> BatchNorm2d<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &BatchNorm2dConfig) -> Self {
        let gamma = Tensor::ones([config.num_features]);
        let beta = Tensor::zeros([config.num_features]);

        let running_mean = Tensor::zeros([config.num_features]);
        let running_var = Tensor::ones([config.num_features]);

        Self {
            gamma: Param::new(gamma),
            beta: Param::new(beta),
            running_mean: Param::new(RunningState::new(running_mean)),
            running_var: Param::new(RunningState::new(running_var)),
            epsilon: config.epsilon,
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, height, width]`
    /// - output: `[batch_size, channels, height, width]`
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = input.dims();
        let squeezed_shape = [batch_size, channels, 1, 1];

        let input_squeezed = input.reshape([batch_size, channels, height * width]);

        let (var, mean) = input_squeezed.var_mean_bias(2);
        let var = var.reshape(squeezed_shape);
        let mean = mean.reshape(squeezed_shape);

        let input_normalized = input.sub(&mean).div(&var.sqrt().add_scalar(self.epsilon));

        input_normalized
            .mul(&self.gamma.reshape(squeezed_shape))
            .add(&self.beta.reshape(squeezed_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestADBackend, TestBackend};
    use burn_tensor::Data;

    #[test]
    fn batch_norm_2d_forward() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestBackend>::new(&config);

        let input = Tensor::from_data(Data::from([[
            [[0.2003, 0.2221], [0.7736, 0.8308]],
            [[0.9154, 0.6224], [0.3819, 0.9818]],
            [[0.8394, 0.7470], [0.7615, 0.7317]],
        ]]));

        let output = module.forward(input);

        output.to_data().assert_approx_eq(
            &Data::from([[
                [[-1.0341, -0.9605], [0.9008, 1.0938]],
                [[0.7916, -0.4290], [-1.4309, 1.0682]],
                [[1.6704, -0.5504], [-0.2019, -0.9181]],
            ]]),
            2,
        );
    }
}
