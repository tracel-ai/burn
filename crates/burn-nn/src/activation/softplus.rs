use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation::softplus;
use burn::tensor::backend::Backend;

/// Softplus layer.
///
/// Applies the softplus function element-wise:
/// `softplus(x) = (1/beta) * log(1 + exp(beta * x))`
///
/// Should be created with [SoftplusConfig](SoftplusConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Softplus {
    /// The beta value.
    pub beta: f64,
}

/// Configuration to create a [Softplus](Softplus) layer using the [init function](SoftplusConfig::init).
#[derive(Config, Debug)]
pub struct SoftplusConfig {
    /// The beta value. Default is 1.0
    #[config(default = "1.0")]
    pub beta: f64,
}

impl SoftplusConfig {
    /// Initialize a new [Softplus](Softplus) Layer
    pub fn init(&self) -> Softplus {
        Softplus { beta: self.beta }
    }
}

impl ModuleDisplay for Softplus {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("beta", &self.beta).optional()
    }
}

impl Softplus {
    /// Forward pass for the Softplus layer.
    ///
    /// See [softplus](burn::tensor::activation::softplus) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        softplus(input, self.beta)
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_softplus_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model: Softplus = SoftplusConfig::new().init();
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0, 1.0, -1.0]]), &device);
        let out = model.forward(input);
        // softplus(0) = log(2) ≈ 0.6931
        // softplus(1) = log(1 + e) ≈ 1.3133
        // softplus(-1) = log(1 + e^-1) ≈ 0.3133
        let expected = TensorData::from([[0.6931, 1.3133, 0.3133]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_softplus_with_beta() {
        let device: burn_ndarray::NdArrayDevice = <TestBackend as Backend>::Device::default();
        let model: Softplus = SoftplusConfig::new().with_beta(2.0).init();
        let input = Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0, 1.0]]), &device);
        let out = model.forward(input);
        // softplus(0, beta=2) = (1/2) * log(1 + exp(0)) = 0.5 * log(2) ≈ 0.3466
        // softplus(1, beta=2) = (1/2) * log(1 + exp(2)) = 0.5 * log(8.389) ≈ 1.0635
        let expected = TensorData::from([[0.3466, 1.0635]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = SoftplusConfig::new().init();
        assert_eq!(alloc::format!("{config}"), "Softplus {beta: 1}");
    }
}
