use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation::softplus;

/// Softplus layer.
///
/// Applies the softplus function element-wise:
/// `softplus(x) = (1/beta) * log(1 + exp(beta * x))`
///
/// Should be created with [SoftplusConfig](SoftplusConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Softplus {
    /// The beta value.
    pub beta: f64,
    /// The stability threshold.
    pub threshold: f64,
}

/// Configuration to create a [Softplus](Softplus) layer using the [init function](SoftplusConfig::init).
#[derive(Config, Debug)]
pub struct SoftplusConfig {
    /// The beta value. Default is 1.0
    #[config(default = "1.0")]
    pub beta: f64,
    /// The value of `beta * x` above which the result is taken to be `x` directly, which is
    /// what keeps `exp` from overflowing. Default is 20.0
    ///
    /// See [softplus](burn::tensor::activation::softplus) for the range of values that make
    /// sense here.
    #[config(default = "20.0")]
    pub threshold: f64,
}

impl SoftplusConfig {
    /// Initialize a new [Softplus](Softplus) Layer
    pub fn init(&self) -> Softplus {
        Softplus {
            beta: self.beta,
            threshold: self.threshold,
        }
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
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        softplus(input, self.beta, self.threshold)
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_softplus_forward() {
        let device = Default::default();
        let model = SoftplusConfig::new().init();
        let input = Tensor::<2>::from_data(TensorData::from([[0.0, 1.0, -1.0]]), &device);
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
        let device = Default::default();
        let model = SoftplusConfig::new().with_beta(2.0).init();
        let input = Tensor::<2>::from_data(TensorData::from([[0.0, 1.0]]), &device);
        let out = model.forward(input);
        // softplus(0, beta=2) = (1/2) * log(1 + exp(0)) = 0.5 * log(2) ≈ 0.3466
        // softplus(1, beta=2) = (1/2) * log(1 + exp(2)) = 0.5 * log(8.389) ≈ 1.0635
        let expected = TensorData::from([[0.3466, 1.0635]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_softplus_default_threshold() {
        let device = Default::default();
        let model = SoftplusConfig::new().init();
        assert_eq!(model.threshold, 20.0);

        // The default threshold keeps saturated inputs finite.
        let input = Tensor::<2>::from_data(TensorData::from([[100.0, 1000.0]]), &device);
        let out = model.forward(input);
        let expected = TensorData::from([[100.0, 1000.0]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_softplus_with_threshold() {
        let device = Default::default();
        let input = Tensor::<2>::from_data(TensorData::from([[5.0]]), &device);

        // Below the threshold the curve is evaluated: softplus(5) = log(1 + e^5) ≈ 5.0067.
        let out = SoftplusConfig::new()
            .with_threshold(10.0)
            .init()
            .forward(input.clone());
        out.to_data()
            .assert_approx_eq::<FT>(&TensorData::from([[5.0067153]]), Tolerance::default());

        // Above it, the identity is substituted instead.
        let out = SoftplusConfig::new()
            .with_threshold(1.0)
            .init()
            .forward(input);
        out.to_data()
            .assert_approx_eq::<FT>(&TensorData::from([[5.0]]), Tolerance::default());
    }

    #[test]
    fn display() {
        let config = SoftplusConfig::new().init();
        assert_eq!(alloc::format!("{config}"), "Softplus {beta: 1}");
    }
}
