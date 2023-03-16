use alloc::{format, vec::Vec};

use crate as burn;

use crate::{
    config::Config,
    module::{Module, Param, RunningState},
    tensor::{backend::Backend, Tensor},
};

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
            gamma: Param::from(gamma),
            beta: Param::from(beta),
            running_mean: Param::from(RunningState::new(running_mean)),
            running_var: Param::from(RunningState::new(running_var)),
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
        let channels = input.dims()[1];
        let mean = self.running_mean.val().value();
        let var = self.running_var.val().value();

        self.forward_shared(
            input,
            mean.reshape([1, channels, 1, 1]),
            var.reshape([1, channels, 1, 1]),
        )
    }

    fn forward_train(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = input.dims();

        let mean = input
            .clone()
            .swap_dims(0, 1)
            .reshape([channels, batch_size * height * width])
            .mean_dim(1)
            .reshape([1, channels, 1, 1]);

        let var = input
            .clone()
            .sub(mean.clone())
            .powf(2.0)
            .swap_dims(0, 1)
            .reshape([channels, batch_size * height * width])
            .mean_dim(1)
            .reshape([1, channels, 1, 1]);

        let running_mean = self.running_mean.value_sync();
        let running_var = self.running_var.value_sync();

        let running_mean = running_mean.mul_scalar(1.0 - self.momentum).add(
            mean.clone()
                .detach()
                .mul_scalar(self.momentum)
                .reshape([channels]),
        );
        let running_var = running_var.mul_scalar(1.0 - self.momentum).add(
            var.clone()
                .detach()
                .mul_scalar(self.momentum)
                .reshape([channels]),
        );

        self.running_mean.update(running_mean.detach());
        self.running_var.update(running_var.detach());

        self.forward_shared(input, mean, var)
    }

    fn forward_shared(
        &self,
        x: Tensor<B, 4>,
        mean: Tensor<B, 4>,
        var: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let channels = x.dims()[1];
        let std = var.add_scalar(self.epsilon).sqrt();

        let x = x.sub(mean);
        let x = x.div(std);

        let x = x.mul(self.gamma.val().reshape([1, channels, 1, 1]));

        x.add(self.beta.val().reshape([1, channels, 1, 1]))
    }
}

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
            &Data::from([
                [
                    [[1.5136, 0.7506], [-1.2216, 0.1477]],
                    [[0.3135, 1.2252], [-0.4150, 0.6130]],
                    [[1.4186, 0.3372], [-1.5183, 1.5262]],
                ],
                [
                    [[0.4483, -1.1914], [-1.2010, 0.7537]],
                    [[-1.6752, 1.3822], [-0.5058, -0.9381]],
                    [[0.0200, -0.3097], [-0.5715, -0.9026]],
                ],
            ]),
            2,
        );
    }

    #[test]
    fn batch_norm_2d_forward_inference() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);

        module.forward(input_tensor());
        let module = module.inner();
        let output = module.forward(input_tensor());

        output.to_data().assert_approx_eq(
            &Data::from([
                [
                    [[0.9538, 0.7103], [0.0808, 0.5179]],
                    [[0.6015, 0.8910], [0.3703, 0.6966]],
                    [[0.9171, 0.6912], [0.3037, 0.9395]],
                ],
                [
                    [[0.6138, 0.0904], [0.0874, 0.7113]],
                    [[-0.0297, 0.9408], [0.3415, 0.2042]],
                    [[0.6250, 0.5561], [0.5013, 0.4323]],
                ],
            ]),
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
            .into_data()
            .assert_approx_eq(&Data::from([0.0499, 0.0532, 0.0656]), 2);
    }

    #[test]
    fn batch_norm_2d_running_var() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);

        let _output = module.forward(input_tensor());

        let running_var = module.running_var.value_sync();

        running_var
            .reshape([3])
            .into_data()
            .assert_approx_eq(&Data::from([0.9106, 0.9105, 0.9045]), 2);
    }

    #[test]
    fn batch_norm_2d_running_mean_inner_module() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);

        let _output = module.forward(input_tensor());

        let module_valid = module.inner();
        let running_mean = module_valid.running_mean.value();

        let module_train = BatchNorm2d::<TestADBackend>::from_inner(module_valid);
        let running_mean_after = module_train.running_mean.value();

        running_mean_after
            .into_data()
            .assert_approx_eq(&running_mean.into_data(), 3);
    }

    #[test]
    fn batch_norm_2d_grads() {
        let config = BatchNorm2dConfig::new(3);
        let module = BatchNorm2d::<TestADBackend>::new(&config);
        let input = input_tensor().require_grad();

        let output = module.forward(input.clone());

        let grads = output.backward();

        module
            .gamma
            .grad(&grads)
            .unwrap()
            .reshape([3])
            .into_data()
            .assert_approx_eq(&Data::from([0.0000e+00, -5.9035e-07, -6.0011e-07]), 3);

        module
            .beta
            .grad(&grads)
            .unwrap()
            .reshape([3])
            .into_data()
            .assert_approx_eq(&Data::from([8., 8., 8.]), 3);

        input.grad(&grads).unwrap().into_data().assert_approx_eq(
            &Data::from([
                [
                    [[0.0000e+00, 0.0000e+00], [0.0000e+00, 0.0000e+00]],
                    [[7.6400e-08, 2.9848e-07], [-1.0110e-07, 1.4933e-07]],
                    [[5.3570e-07, 1.2732e-07], [-5.7336e-07, 5.7632e-07]],
                ],
                [
                    [[0.0000e+00, 0.0000e+00], [0.0000e+00, 0.0000e+00]],
                    [[-4.0807e-07, 3.3673e-07], [-1.2323e-07, -2.2854e-07]],
                    [[7.5642e-09, -1.1695e-07], [-2.1582e-07, -3.4078e-07]],
                ],
            ]),
            4,
        );
    }

    fn input_tensor<B: Backend>() -> Tensor<B, 4> {
        Tensor::<B, 4>::from_floats([
            [
                [[0.9601, 0.7277], [0.1270, 0.5441]],
                [[0.6272, 0.9034], [0.4066, 0.7179]],
                [[0.9378, 0.7230], [0.3544, 0.9591]],
            ],
            [
                [[0.6356, 0.1362], [0.1333, 0.7287]],
                [[0.0249, 0.9509], [0.3791, 0.2481]],
                [[0.6600, 0.5945], [0.5424, 0.4767]],
            ],
        ])
    }
}
