use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation::celu;
use burn::tensor::backend::Backend;

/// CELU (Continuously Differentiable Exponential Linear Unit) layer.
///
/// Applies the CELU function element-wise:
/// `celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`
///
/// Should be created with [CeluConfig](CeluConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Celu {
    /// The alpha value for the CELU formulation.
    pub alpha: f64,
}

/// Configuration to create a [Celu](Celu) layer using the [init function](CeluConfig::init).
#[derive(Config, Debug)]
pub struct CeluConfig {
    /// The alpha value for the CELU formulation. Default is 1.0
    #[config(default = "1.0")]
    pub alpha: f64,
}

impl CeluConfig {
    /// Initialize a new [Celu](Celu) Layer
    pub fn init(&self) -> Celu {
        Celu { alpha: self.alpha }
    }
}

impl ModuleDisplay for Celu {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("alpha", &self.alpha).optional()
    }
}

impl Celu {
    /// Forward pass for the Celu layer.
    ///
    /// See [celu](burn::tensor::activation::celu) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        celu(input, self.alpha)
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
    fn test_celu_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model: Celu = CeluConfig::new().init();
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.5, -0.5, -1.0]]), &device);
        let out = model.forward(input);
        // celu(0.5, 1) = 0.5
        // celu(-0.5, 1) = 1 * (exp(-0.5) - 1) = -0.393469
        // celu(-1.0, 1) = 1 * (exp(-1) - 1) = -0.632121
        let expected = TensorData::from([[0.5, -0.393469, -0.632121]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_celu_with_alpha() {
        let device = <TestBackend as Backend>::Device::default();
        let model: Celu = CeluConfig::new().with_alpha(2.0).init();
        let input = Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0, -2.0]]), &device);
        let out = model.forward(input);
        // celu(0, 2) = 0
        // celu(-2, 2) = 2 * (exp(-1) - 1) = -1.264241
        let expected = TensorData::from([[0.0, -1.264241]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = CeluConfig::new().init();
        assert_eq!(alloc::format!("{config}"), "Celu {alpha: 1}");
    }
}
