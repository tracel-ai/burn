use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_core as burn;

use burn::tensor::activation::elu;

/// ELU (Exponential Linear Unit) layer.
///
/// Should be created with [EluConfig](EluConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Elu {
    /// The alpha value.
    pub alpha: f64,
}
/// Configuration to create an [Elu](Elu) layer using the [init function](EluConfig::init).
#[derive(Config, Debug)]
pub struct EluConfig {
    /// The alpha value. Default is 1.0
    #[config(default = "1.0")]
    pub alpha: f64,
}
impl EluConfig {
    /// Initialize a new [Elu](Elu) Layer
    pub fn init(&self) -> Elu {
        Elu { alpha: self.alpha }
    }
}

impl ModuleDisplay for Elu {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("alpha", &self.alpha).optional()
    }
}

impl Elu {
    /// Forward pass for the ELU layer.
    ///
    /// See [elu](burn::tensor::activation::elu) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        elu(input, self.alpha)
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
    fn test_elu_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model: Elu = EluConfig::new().init();
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.4410, -0.2507]]), &device);
        let out = model.forward(input);
        // elu(0.4410, 1.0) = 0.4410
        // elu(-0.2507, 1.0) = 1.0 * (exp(-0.2507) - 1) = -0.22186
        let expected = TensorData::from([[0.4410, -0.22186]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = EluConfig::new().init();
        assert_eq!(alloc::format!("{config}"), "Elu {alpha: 1}");
    }
}
