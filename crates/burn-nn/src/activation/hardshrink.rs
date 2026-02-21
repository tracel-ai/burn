use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation::hard_shrink;
use burn::tensor::backend::Backend;

/// Hard Shrink layer.
///
/// Applies the Hard Shrink function element-wise:
/// `hard_shrink(x) = x if |x| > lambda else 0`
///
/// Should be created with [HardShrinkConfig](HardShrinkConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct HardShrink {
    /// The lambda value for the Hard Shrink formulation.
    pub lambda: f64,
}

/// Configuration to create a [HardShrink](HardShrink) layer using the [init function](HardShrinkConfig::init).
#[derive(Config, Debug)]
pub struct HardShrinkConfig {
    /// The lambda value for the Hard Shrink formulation. Default is 0.5
    #[config(default = "0.5")]
    pub lambda: f64,
}

impl HardShrinkConfig {
    /// Initialize a new [HardShrink](HardShrink) Layer
    pub fn init(&self) -> HardShrink {
        HardShrink {
            lambda: self.lambda,
        }
    }
}

impl ModuleDisplay for HardShrink {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("lambda", &self.lambda).optional()
    }
}

impl HardShrink {
    /// Forward pass for the Hard Shrink layer.
    ///
    /// See [hard_shrink](burn::tensor::activation::hard_shrink) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        hard_shrink(input, self.lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;

    #[test]
    fn test_hard_shrink_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model: HardShrink = HardShrinkConfig::new().init();
        let input = Tensor::<TestBackend, 2>::from_floats(
            TensorData::from([[0.5, -0.5, -1.0], [8.0, 0.3, 0.0]]),
            &device,
        );
        let out = model.forward(input);
        let expected = TensorData::from([[0.0_f32, 0.0, -1.0], [8.0, 0.0, 0.0]]);
        assert_eq!(out.into_data(), expected);
    }

    #[test]
    fn test_hard_shrink_with_lambda() {
        let device = <TestBackend as Backend>::Device::default();
        let model: HardShrink = HardShrinkConfig::new().with_lambda(0.2).init();
        let input = Tensor::<TestBackend, 2>::from_floats(
            TensorData::from([[0.1, -0.1, -0.3], [0.5, 0.1, 0.0]]),
            &device,
        );
        let out = model.forward(input);
        let expected = TensorData::from([[0.0_f32, 0.0, -0.3], [0.5, 0.0, 0.0]]);
        assert_eq!(out.into_data(), expected);
    }

    #[test]
    fn display() {
        let config = HardShrinkConfig::new().init();
        assert_eq!(alloc::format!("{config}"), "HardShrink {lambda: 0.5}");
    }
}
