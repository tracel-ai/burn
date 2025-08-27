use crate as burn;
use crate::module::{Content, DisplaySettings, ModuleDisplay};

use crate::nn::Initializer;
use crate::{
    config::Config,
    module::{Module, Param, RunningState},
    tensor::{Tensor, backend::Backend},
};

/// [`BatchNorm`] Configuration.
///
/// Used to create a [`BatchNorm`] layer using the [`BatchNormConfig::init`].
#[derive(Config, Debug)]
pub struct BatchNormConfig {
    /// The number of features.
    pub num_features: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// Momentum used to update the metrics. Default: 0.1
    #[config(default = 0.1)]
    pub momentum: f64,
}

/// Applies Batch Normalization over a tensor.
///
/// Based upon the paper [Batch Normalization](https://arxiv.org/abs/1502.03167).
///
/// Assumes input tensor is of shape ``[batch_size, channels, ...]``.
///
/// `Y = norm(X) * γ + β`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `norm` is the normalization function
/// - `γ` is the learnable weight
/// - `β` is the learnable bias
///
/// Should be created using [`BatchNormConfig`].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct BatchNorm<B: Backend> {
    /// The learnable weight gamma.
    pub gamma: Param<Tensor<B, 1>>,
    /// The learnable weight beta.
    pub beta: Param<Tensor<B, 1>>,
    /// The running mean.
    pub running_mean: RunningState<Tensor<B, 1>>,
    /// The running variance.
    pub running_var: RunningState<Tensor<B, 1>>,
    /// Momentum used to update the metrics.
    pub momentum: f64,
    /// A value required for numerical stability.
    pub epsilon: f64,
}

impl BatchNormConfig {
    /// Initializes a new [batch norm](BatchNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BatchNorm<B> {
        let gamma = Initializer::Ones.init([self.num_features], device);
        let beta = Initializer::Zeros.init([self.num_features], device);

        let running_mean = Tensor::zeros([self.num_features], device);
        let running_var = Tensor::ones([self.num_features], device);

        BatchNorm {
            gamma,
            beta,
            running_mean: RunningState::new(running_mean),
            running_var: RunningState::new(running_var),
            momentum: self.momentum,
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> BatchNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [`BatchNorm`] for more information.
    ///
    /// # Shapes
    ///
    /// - `input`: ``[batch_size, channels, ...]``
    /// - `output`: ``[batch_size, channels, ...]``
    ///
    /// # Panics
    ///
    /// This function will panic if the input tensor has rank < 2.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Should be move to a compilation error when const generic support that kind of
        // validation. https://github.com/rust-lang/rust/issues/76560
        if D < 2 {
            panic!(
                "BatchNorm can only be applied on tensors of rank >= 2 with the following shape \
                 [batch_size, channels, ...], received {}D tensor",
                D
            );
        }

        match B::ad_enabled() {
            true => self.forward_train(input),
            false => self.forward_inference(input),
        }
    }

    fn forward_inference<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let device = input.device();
        let channels = input.dims()[1];
        let mean = self.running_mean.value().to_device(&device);
        let var = self.running_var.value().to_device(&device);

        let mut shape = [1; D];
        shape[1] = channels;

        self.forward_shared(input, mean.reshape(shape), var.reshape(shape))
    }

    fn forward_train<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let device = input.device();
        let dims = input.dims();
        let batch_size = dims[0];
        let channels = dims[1];

        let mut shape_unsqueeze = [1; D];
        let mut flatten_size = batch_size;
        shape_unsqueeze[1] = channels;

        for dim in dims.iter().take(D).skip(2) {
            flatten_size *= dim;
        }

        let mean = input
            .clone()
            .swap_dims(0, 1)
            .reshape([channels, flatten_size])
            .mean_dim(1)
            .reshape(shape_unsqueeze);

        let var = input
            .clone()
            .sub(mean.clone())
            .powi_scalar(2)
            .swap_dims(0, 1)
            .reshape([channels, flatten_size])
            .mean_dim(1)
            .reshape(shape_unsqueeze);

        let running_mean = self.running_mean.value_sync().to_device(&device);
        let running_var = self.running_var.value_sync().to_device(&device);

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

    fn forward_shared<const D: usize>(
        &self,
        x: Tensor<B, D>,
        mean: Tensor<B, D>,
        var: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let channels = x.dims()[1];
        let mut shape = [1; D];
        shape[1] = channels;

        let std = var.add_scalar(self.epsilon).sqrt();

        let x = x.sub(mean);
        let x = x.div(std);

        let x = x.mul(self.gamma.val().reshape(shape));

        x.add(self.beta.val().reshape(shape))
    }
}

impl<B: Backend> ModuleDisplay for BatchNorm<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [num_features] = self.beta.shape().dims();

        content
            .add("num_features", &num_features)
            .add("momentum", &self.momentum)
            .add("epsilon", &self.epsilon)
            .optional()
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests_1d {
    use super::*;
    use crate::tensor::TensorData;
    use crate::{TestAutodiffBackend, module::AutodiffModule};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestAutodiffBackend>;

    #[test]
    fn batch_norm_forward_train() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);

        let output = module.forward(input_tensor(&device));

        let expected = TensorData::from([
            [
                [1.1483e+00, 3.7521e-01],
                [1.6272e-03, 7.5067e-01],
                [1.6204e+00, -4.5168e-02],
            ],
            [
                [6.8856e-02, -1.5923e+00],
                [-1.6318e+00, 8.7949e-01],
                [-5.3368e-01, -1.0416e+00],
            ],
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(0.1, 0.001));
    }

    #[test]
    fn batch_norm_forward_inference() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);

        module.forward(input_tensor(&device));
        let module = module.valid();
        let output = module.forward(input_tensor(&device));

        let expected = TensorData::from([
            [[0.9409, 0.6976], [0.5892, 0.8774], [0.9106, 0.6844]],
            [[0.6012, 0.0782], [-0.0394, 0.9270], [0.6181, 0.5492]],
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    fn input_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 3> {
        Tensor::<B, 3>::from_floats(
            [
                [[0.9601, 0.7277], [0.6272, 0.9034], [0.9378, 0.7230]],
                [[0.6356, 0.1362], [0.0249, 0.9509], [0.6600, 0.5945]],
            ],
            device,
        )
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests_2d {
    use super::*;
    use crate::tensor::TensorData;
    use crate::{TestAutodiffBackend, module::AutodiffModule};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestAutodiffBackend>;

    #[test]
    fn batch_norm_forward_train() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);

        let output = module.forward(input_tensor(&device));

        let expected = TensorData::from([
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
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(0.1, 0.001));
    }

    #[test]
    fn batch_norm_forward_inference() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);

        module.forward(input_tensor(&device));
        let module = module.valid();
        let output = module.forward(input_tensor(&device));

        let expected = TensorData::from([
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
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn batch_norm_running_mean() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);

        let _output = module.forward(input_tensor(&device));

        let running_mean = module.running_mean.value_sync();

        let expected = TensorData::from([0.0499, 0.0532, 0.0656]);
        running_mean
            .reshape([3])
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn batch_norm_running_var() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);

        let _output = module.forward(input_tensor(&device));

        let running_var = module.running_var.value_sync();

        let expected = TensorData::from([0.9106, 0.9105, 0.9045]);
        running_var
            .reshape([3])
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn batch_norm_running_mean_inner_module() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);

        let _output = module.forward(input_tensor(&device));

        let module_valid = module.valid();
        let running_mean = module_valid.running_mean.value();
        let running_mean_after = module.running_mean.value();

        running_mean_after
            .into_data()
            .assert_approx_eq::<FT>(&running_mean.into_data(), Tolerance::default());
    }

    #[test]
    fn batch_norm_grads() {
        let device = Default::default();
        let module = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);
        let input = input_tensor(&device).require_grad();

        let output = module.forward(input.clone());

        let grads = output.backward();

        let tolerance = Tolerance::rel_abs(0.1, 0.001);
        let expected = TensorData::from([0.0000e+00, -5.9035e-07, -6.0011e-07]);
        module
            .gamma
            .grad(&grads)
            .unwrap()
            .reshape([3])
            .into_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        let expected = TensorData::from([8., 8., 8.]);
        module
            .beta
            .grad(&grads)
            .unwrap()
            .reshape([3])
            .into_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        let expected = TensorData::from([
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
        ]);
        input
            .grad(&grads)
            .unwrap()
            .into_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
    }

    fn input_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 4> {
        Tensor::<B, 4>::from_floats(
            [
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
            ],
            device,
        )
    }

    #[test]
    fn display() {
        let batch_norm = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&Default::default());

        assert_eq!(
            format!("{batch_norm}"),
            "BatchNorm {num_features: 3, momentum: 0.1, epsilon: 0.00001, params: 12}"
        );
    }
}
