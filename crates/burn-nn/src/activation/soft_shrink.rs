use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation::soft_shrink;
use burn::tensor::backend::Backend;

/// Soft Shrink layer.
///
/// Applies the Soft Shrink function element-wise:
/// `soft_shrink(x) = x - bias if x > lambda, x + bias if x < -lambda, 0 otherwise`
///
/// Should be created with [SoftShrinkConfig](SoftShrinkConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct SoftShrink {
    /// The lambda value for the Soft Shrink formulation.
    pub lambda: f64,
    /// The bias value for the Soft Shrink formulation.
    // Usually bias = lambda, but need this to handle onnx spec https://onnx.ai/onnx/operators/onnx__Shrink.html
    pub bias: f64,
}

/// Configuration to create a [SoftShrink](SoftShrink) layer using the [init function](SoftShrinkConfig::init).
#[derive(Config, Debug)]
pub struct SoftShrinkConfig {
    /// The lambda value for the Soft Shrink formulation. Default is 0.5
    #[config(default = "0.5")]
    pub lambda: f64,
    /// The bias value for the Soft Shrink formulation. Default is 0.5.
    #[config(default = "0.5")]
    pub bias: f64,
}

impl SoftShrinkConfig {
    /// Initialize a new [SoftShrink](SoftShrink) Layer
    pub fn init(&self) -> SoftShrink {
        SoftShrink {
            lambda: self.lambda,
            bias: self.bias,
        }
    }
}

impl ModuleDisplay for SoftShrink {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("lambda", &self.lambda)
            .add("bias", &self.bias)
            .optional()
    }
}

impl SoftShrink {
    /// Forward pass for the Soft Shrink layer.
    ///
    /// See [soft_shrink](burn::tensor::activation::soft_shrink) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        soft_shrink(input, self.lambda, self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;

    #[test]
    fn test_soft_shrink_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model: SoftShrink = SoftShrinkConfig::new().init();
        let input =
            Tensor::<TestBackend, 2>::from_data([[0.5, -0.5, -1.0], [8.0, 0.3, 0.0]], &device);
        let out = model.forward(input);
        let expected = TensorData::from([[0.0_f32, 0.0, -0.5], [7.5, 0.0, 0.0]]);
        assert_eq!(out.into_data(), expected);
    }

    #[test]
    fn test_soft_shrink_with_lambda_and_bias() {
        let device = <TestBackend as Backend>::Device::default();
        let model: SoftShrink = SoftShrinkConfig::new()
            .with_lambda(0.25)
            .with_bias(0.125)
            .init();
        let input =
            Tensor::<TestBackend, 2>::from_data([[0.125, -0.125, -0.5], [0.75, 0.1, 0.0]], &device);
        let out = model.forward(input);
        let expected = TensorData::from([[0.0_f32, 0.0, -0.375], [0.625, 0.0, 0.0]]);
        assert_eq!(out.into_data(), expected);
    }

    #[test]
    fn display() {
        let config = SoftShrinkConfig::new().init();
        assert_eq!(
            alloc::format!("{config}"),
            "SoftShrink {lambda: 0.5, bias: 0.5}"
        );
    }
}
