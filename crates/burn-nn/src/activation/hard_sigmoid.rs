use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation::hard_sigmoid;
use burn::tensor::backend::Backend;

/// Hard Sigmoid layer.
///
/// Should be created with [HardSigmoidConfig](HardSigmoidConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct HardSigmoid {
    /// The alpha value.
    pub alpha: f64,
    /// The beta value.
    pub beta: f64,
}
/// Configuration to create a [Hard Sigmoid](HardSigmoid) layer using the [init function](HardSigmoidConfig::init).
#[derive(Config, Debug)]
pub struct HardSigmoidConfig {
    /// The alpha value. Default is 0.2
    #[config(default = "0.2")]
    pub alpha: f64,
    /// The beta value. Default is 0.5
    #[config(default = "0.5")]
    pub beta: f64,
}
impl HardSigmoidConfig {
    /// Initialize a new [Hard Sigmoid](HardSigmoid) Layer
    pub fn init(&self) -> HardSigmoid {
        HardSigmoid {
            alpha: self.alpha,
            beta: self.beta,
        }
    }
}

impl ModuleDisplay for HardSigmoid {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("alpha", &self.alpha)
            .add("beta", &self.beta)
            .optional()
    }
}

impl HardSigmoid {
    /// Forward pass for the Hard Sigmoid layer.
    ///
    /// See [hard_sigmoid](burn::tensor::activation::hard_sigmoid) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        hard_sigmoid(input, self.alpha, self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_hard_sigmoid_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model: HardSigmoid = HardSigmoidConfig::new().init();
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.4410, -0.2507]]), &device);
        let out = model.forward(input);
        let expected = TensorData::from([[0.5882, 0.44986]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = HardSigmoidConfig::new().init();
        assert_eq!(
            alloc::format!("{config}"),
            "HardSigmoid {alpha: 0.2, beta: 0.5}"
        );
    }
}
