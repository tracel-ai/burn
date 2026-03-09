use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

/// Configuration to create a [Dropout](Dropout) layer using the [init function](DropoutConfig::init).
#[derive(Config, Debug)]
pub struct DropoutConfig {
    /// The probability of randomly zeroes some elements of the input tensor during training.
    pub prob: f64,
}

/// Set at random some elements of the input tensor to zero during training.
///
/// This is an effective regularization technique as describe in the paper
/// [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580).
///
/// The input is also scaled during training to `1 / (1 - prob_keep)`.
///
/// Should be created with [DropoutConfig].
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Dropout {
    /// The probability of randomly zeroes some elements of the input tensor during training.
    pub prob: f64,
}

impl DropoutConfig {
    /// Initialize a new [dropout](Dropout) module.
    pub fn init(&self) -> Dropout {
        if self.prob < 0.0 || self.prob > 1.0 {
            panic!(
                "Dropout probability should be between 0 and 1, but got {}",
                self.prob
            );
        }
        Dropout { prob: self.prob }
    }
}

impl Dropout {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [Dropout](Dropout) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if !B::ad_enabled(&input.device()) || self.prob == 0.0 {
            return input;
        }

        let prob_keep = 1.0 - self.prob;
        let random = input.random_like(Distribution::Bernoulli(prob_keep));
        let x = input * random;

        x * (1.0 / prob_keep)
    }
}

impl ModuleDisplay for Dropout {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("prob", &self.prob).optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Shape;

    #[cfg(feature = "std")]
    use crate::{TestAutodiffBackend, TestBackend};

    #[cfg(not(feature = "std"))]
    use crate::TestBackend;

    #[cfg(feature = "std")]
    #[test]
    fn with_ad_backend_should_mark_input() {
        let tensor =
            Tensor::<TestAutodiffBackend, 2>::ones(Shape::new([100, 100]), &Default::default());
        let dropout = DropoutConfig::new(0.5).init();

        let output = dropout.forward(tensor.clone());

        assert_ne!(tensor.to_data(), output.to_data());
    }

    #[test]
    fn without_ad_backend_should_not_change_input() {
        let tensor = Tensor::<TestBackend, 2>::ones(Shape::new([100, 100]), &Default::default());
        let dropout = DropoutConfig::new(0.5).init();

        let output = dropout.forward(tensor.clone());

        assert_eq!(tensor.to_data(), output.to_data());
    }

    #[test]
    fn display() {
        let config = DropoutConfig::new(0.5);
        let layer = config.init();

        assert_eq!(alloc::format!("{layer}"), "Dropout {prob: 0.5}");
    }

    #[test]
    #[should_panic = "Dropout probability should be between 0 and 1,"]
    fn dropout_prob_invalid() {
        let config = DropoutConfig::new(-10.);
        let _layer = config.init();
    }
}
