use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_core as burn;

use burn::tensor::activation::thresholded_relu;

/// Thresholded ReLU layer.
///
/// Should be created with [ThresholdedReluConfig](ThresholdedReluConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct ThresholdedRelu {
    /// The alpha threshold.
    pub alpha: f64,
}

/// Configuration to create a [ThresholdedRelu](ThresholdedRelu) layer using the [init function](ThresholdedReluConfig::init).
#[derive(Config, Debug)]
pub struct ThresholdedReluConfig {
    /// The alpha threshold. Default is 1.0
    #[config(default = "1.0")]
    pub alpha: f64,
}

impl ThresholdedReluConfig {
    /// Initialize a new [ThresholdedRelu](ThresholdedRelu) layer.
    pub fn init(&self) -> ThresholdedRelu {
        ThresholdedRelu { alpha: self.alpha }
    }
}

impl ModuleDisplay for ThresholdedRelu {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("alpha", &self.alpha).optional()
    }
}

impl ThresholdedRelu {
    /// Forward pass for the Thresholded ReLU layer.
    ///
    /// See [thresholded_relu](burn::tensor::activation::thresholded_relu) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        thresholded_relu(input, self.alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;

    #[test]
    fn test_thresholded_relu_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model: ThresholdedRelu = ThresholdedReluConfig::new().init();
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.5, 1.5, -0.2]]), &device);
        let out = model.forward(input);
        let expected = TensorData::from([[0.0, 1.5, 0.0]]);
        out.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn display() {
        let config = ThresholdedReluConfig::new().init();
        assert_eq!(alloc::format!("{config}"), "ThresholdedRelu {alpha: 1}");
    }
}
