use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

/// Configuration to create a [GaussianNoise](GaussianNoise) layer using the [init function](GaussianNoiseConfig::init).
#[derive(Config, Debug)]
pub struct GaussianNoiseConfig {
    /// Standard deviation of the normal noise distribution.
    pub std: f64,
}

/// Add pseudorandom Gaussian noise to an arbitrarily shaped tensor.
///
/// This is an effective regularization technique that also contributes to data augmentation.
/// Please keep in mind that the value of [std](GaussianNoise::std) should be chosen with care in order to avoid
/// distortion.
///
/// Should be created with [GaussianNoiseConfig].
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct GaussianNoise {
    /// Standard deviation of the normal noise distribution.
    pub std: f64,
}

impl GaussianNoiseConfig {
    /// Initialize a new [Gaussian noise](GaussianNoise) module.
    pub fn init(&self) -> GaussianNoise {
        if self.std.is_sign_negative() {
            panic!(
                "Standard deviation is required to be non-negative, but got {}",
                self.std
            );
        }
        GaussianNoise { std: self.std }
    }
}

impl GaussianNoise {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [GaussianNoise](GaussianNoise) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if B::ad_enabled() && self.std != 0.0 {
            let noise = Tensor::random(
                input.shape(),
                Distribution::Normal(0.0, self.std),
                &input.device(),
            );
            input + noise
        } else {
            input
        }
    }
}

impl ModuleDisplay for GaussianNoise {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("std", &self.std).optional()
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
        let noise = GaussianNoiseConfig::new(0.5).init();

        let output = noise.forward(tensor.clone());

        assert_ne!(tensor.to_data(), output.to_data());
    }

    #[test]
    fn without_ad_backend_should_not_change_input() {
        let tensor = Tensor::<TestBackend, 2>::ones(Shape::new([100, 100]), &Default::default());
        let noise = GaussianNoiseConfig::new(0.5).init();

        let output = noise.forward(tensor.clone());

        assert_eq!(tensor.to_data(), output.to_data());
    }

    #[test]
    #[should_panic(expected = "Standard deviation is required to be non-negative")]
    fn negative_std_should_panic() {
        GaussianNoiseConfig { std: -0.5 }.init();
    }

    #[test]
    fn display() {
        let config = GaussianNoiseConfig::new(0.5);
        let layer = config.init();

        assert_eq!(alloc::format!("{layer}"), "GaussianNoise {std: 0.5}");
    }
}
