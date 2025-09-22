use burn::tensor::DType;

use burn_core as burn;

use burn::config::Config;
use burn::module::Initializer;
use burn::module::Module;
use burn::module::Param;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Configuration to create a [RMS Norm](RmsNorm) layer using the [init function](RmsNormConfig::init).
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// The size of the input features.
    pub d_model: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl RmsNormConfig {
    /// Initialize a new [RMS Norm](RmsNorm) module.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` is not positive.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        assert!(self.epsilon > 0.0, "epsilon must be positive.");

        let gamma = Initializer::Ones.init([self.d_model], device);

        RmsNorm {
            gamma,
            epsilon: self.epsilon,
        }
    }
}

/// Applies RMS Normalization over an input tensor along the last dimension.
///
/// `Y = X / sqrt(mean(X^2) + eps) * gamma`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `gamma` is the learnable weight
/// - `mean` is the mean operation
/// - `eps` is a small value to avoid division by zero.
///
/// Should be created using the [RmsNormConfig](RmsNormConfig) configuration.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct RmsNorm<B: Backend> {
    /// The learnable parameter to scale the normalized tensor
    pub gamma: Param<Tensor<B, 1>>,
    /// A value required for numerical stability
    pub epsilon: f64,
}

impl<B: Backend> RmsNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See the [RmsNorm](RmsNorm) documentation for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // Calculate the root-mean-square norm of the input tensor along the last dimension
        let dtype = x.dtype();
        let rms = (x.clone().cast(DType::F32).powi_scalar(2).mean_dim(D - 1) + self.epsilon).sqrt();
        (x / rms.cast(dtype)) * self.gamma.val().unsqueeze()
    }
}

impl<B: Backend> ModuleDisplay for RmsNorm<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_model] = self.gamma.shape().dims();
        content
            .add("d_model", &d_model)
            .add("epsilon", &self.epsilon)
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use alloc::format;
    use burn::tensor::TensorData;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn rms_norm_forward() {
        let device = Default::default();
        let module = RmsNormConfig::new(3)
            .with_epsilon(1e-5)
            .init::<TestBackend>(&device);

        let input = Tensor::arange(0..9, &device).float().reshape([3, 3]);

        let output = module.forward(input);

        let expected = TensorData::from([
            [0.0000, 0.7746, 1.5492],
            [0.7348, 0.9798, 1.2247],
            [0.8514, 0.9933, 1.1352],
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = RmsNormConfig::new(6);
        let layer_norm = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            format!("{layer_norm}"),
            "RmsNorm {d_model: 6, epsilon: 0.00001, params: 6}"
        );
    }
}
