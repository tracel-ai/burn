use std::f32::consts::PI;

use crate as burn;
use crate::module::{Content, DisplaySettings, ModuleDisplay};
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use crate::{config::Config, module::Module};

use super::Reduction;

/// Configuration for creating a [PoissonNLLLoss](PoissonNLLLoss) instance.
///
/// This configuration allows customization of the Poisson Negative Log Likelihood (NLL) loss
/// behavior, such as whether the input is in log-space, whether to include the Stirling
/// approximation term, and a small epsilon value to avoid numerical instability.
#[derive(Config, Debug)]
pub struct PoissonNLLLossConfig {
    /// If `true`, the predictions are expected to be in log-space.
    ///
    /// When `log_input` is `true`, the loss is computed as:
    /// ```text
    /// L(predictions, target) = exp(predictions) - target * predictions
    /// ```
    /// When `log_input` is `false`, the loss is computed as:
    /// ```text
    /// L(predictions, target) = predictions - target * log(predictions + eps)
    /// ```
    #[config(default = true)]
    pub log_input: bool,
    /// Whether to compute the full loss, including the Stirling approximation term.
    ///
    /// When `full` is `true`, the Stirling approximation term is added to the loss:
    /// ```text
    /// target * log(target) - target + 0.5 * log(2 * PI * target)
    /// ```
    #[config(default = false)]
    pub full: bool,
    /// A small value to avoid evaluation of `log(0)` when `log_input` is `false`.
    ///
    /// This epsilon value is added to the predictions to ensure numerical stability
    /// when computing the logarithm.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl PoissonNLLLossConfig {
    /// Initializes a [PoissonNLLLoss](PoissonNLLLoss) instance with the current configuration.
    ///
    /// # Panics
    /// - Panics if `eps` is not a positive number.
    pub fn init(&self) -> PoissonNLLLoss {
        self.assertions();
        PoissonNLLLoss {
            log_input: self.log_input,
            full: self.full,
            eps: self.eps,
        }
    }

    /// Validates the configuration parameters.
    ///
    /// # Panics
    /// - Panics if `eps` is not a positive number.
    fn assertions(&self) {
        assert!(
            self.eps > 0.,
            "eps for PoissonNLLLoss must be a positive number."
        );
    }
}

