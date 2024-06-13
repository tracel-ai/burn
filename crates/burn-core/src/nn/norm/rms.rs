use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Configuration to create a [RMS Norm](RmsNorm) layer using the [init function](RmsNormConfig::init).
#[derive(Config)]
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
pub struct RmsNorm<B: Backend> {
    /// The learnable parameter to scale the normalized tensor
    pub gamma: Param<Tensor<B, 1>>,
    /// A value required for numerical stability
    epsilon: f64,
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
        let rms = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + self.epsilon).sqrt();
        (x / rms) * self.gamma.val().unsqueeze()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Data;
    use crate::TestBackend;

    #[test]
    fn rms_norm_forward() {
        let device = Default::default();
        let module = RmsNormConfig::new(3)
            .with_epsilon(1e-5)
            .init::<TestBackend>(&device);

        let input = Tensor::arange(0..9, &device).float().reshape([3, 3]);

        let output = module.forward(input);

        output.to_data().assert_approx_eq(
            &Data::from([
                [0.0000, 0.7746, 1.5492],
                [0.7348, 0.9798, 1.2247],
                [0.8514, 0.9933, 1.1352],
            ]),
            4,
        );
    }
}
