use alloc::{format, vec::Vec};

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
    momentum: f64,
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
            momentum: config.momentum,
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
        match B::ad_enabled() {
            true => self.forward_train(input),
            false => self.forward_inference(input),
        }
    }

    fn forward_inference(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, channels, _height, _width] = input.dims();

        let mean = self.running_mean.value();
        let var = self.running_var.value();

        self.forward_shared(
            input,
            mean.reshape([1, channels, 1, 1]),
            var.reshape([1, channels, 1, 1]),
        )
    }

    fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, channels, _height, _width] = input.dims();

        let mean = input.clone().mean_dim(3).mean_dim(2);
        let var = input
            .clone()
            .sub(mean.clone())
            .powf(2.0)
            .mean_dim(3)
            .mean_dim(2);

        let running_mean = self.running_mean.value_sync();
        let running_var = self.running_var.value_sync();

        let running_mean = running_mean.mul_scalar(1.0 - self.momentum).add(
            mean.clone()
                .mean_dim(0)
                .mul_scalar(self.momentum)
                .reshape([channels]),
        );
        let running_var = running_var.mul_scalar(1.0 - self.momentum).add(
            var.clone()
                .mean_dim(0)
                .mul_scalar(self.momentum)
                .reshape([channels]),
        );

        self.running_mean.update(running_mean.detach());
        self.running_var.update(running_var.detach());

        self.forward_shared(input, mean, var)
    }

    fn forward_shared(
        &self,
        input: Tensor<B, 4>,
        mean: Tensor<B, 4>,
        var: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [_batch_size, channels, _, _] = input.dims();
        let input_normalized = input.sub(mean).div(var.sqrt().add_scalar(self.epsilon));

        input_normalized
            .mul(self.gamma.val().reshape([1, channels, 1, 1]))
            .add(self.beta.val().reshape([1, channels, 1, 1]))
    }
}

// TODO test with regular TestBackend

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{module::ADModule, TestADBackend};
    use burn_tensor::Data;

    #[test]
    fn batch_norm_2d_forward_train() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);

        let output = module.forward(input_tensor());

        output.to_data().assert_approx_eq(
            &Data::from([[
                [[-1.0341, -0.9605], [0.9008, 1.0938]],
                [[0.7916, -0.4290], [-1.4309, 1.0682]],
                [[1.6704, -0.5504], [-0.2019, -0.9181]],
            ]]),
            2,
        );
    }

    #[test]
    fn batch_norm_2d_forward_inference() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);

        module.forward(input_tensor());
        let output = module.inner().forward(input_tensor());

        output.to_data().assert_approx_eq(
            &Data::from([[
                [[0.1569, 0.1795], [0.7571, 0.8170]],
                [[0.8851, 0.5771], [0.3251, 0.9551]],
                [[0.8035, 0.7062], [0.7214, 0.6900]],
            ]]),
            2,
        );
    }

    #[test]
    fn batch_norm_2d_running_mean() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);

        let _output = module.forward(input_tensor());

        let running_mean = module.running_mean.value_sync();

        running_mean
            .reshape([3])
            .to_data()
            .assert_approx_eq(&Data::from([0.0507, 0.0725, 0.0770]), 2);
    }

    #[test]
    fn batch_norm_2d_running_var() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);

        let _output = module.forward(input_tensor());

        let running_var = module.running_var.value_sync();

        running_var
            .reshape([3])
            .to_data()
            .assert_approx_eq(&Data::from([0.9117, 0.9077, 0.9002]), 2);
    }

    #[test]
    fn batch_norm_2d_grads() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);
        let input = input_tensor();

        let output = module.forward(input.clone());

        let grads = output.backward();

        module
            .gamma
            .grad(&grads)
            .unwrap()
            .into_data()
            .assert_approx_eq(&Data::from([2.0116e-07, 2.4831e-07, 2.8651e-06]), 3);

        module
            .beta
            .grad(&grads)
            .unwrap()
            .into_data()
            .assert_approx_eq(&Data::from([4., 4., 4.]), 3);

        input.grad(&grads).unwrap().into_data().assert_approx_eq(
            &Data::from([[
                [[1.7550e-07, 1.6301e-07], [-1.5288e-07, -1.8564e-07]],
                [[-2.0472e-07, 1.1094e-07], [3.7004e-07, -2.7626e-07]],
                [[-2.8757e-05, 9.4754e-06], [3.4757e-06, 1.5806e-05]],
            ]]),
            4,
        );
    }

    fn input_tensor<B: Backend>() -> Tensor<B, 4> {
        Tensor::<B, 4>::from_floats([[
            [[0.2003, 0.2221], [0.7736, 0.8308]],
            [[0.9154, 0.6224], [0.3819, 0.9818]],
            [[0.8394, 0.7470], [0.7615, 0.7317]],
        ]])
    }
}
