use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn_core as burn;

use burn::tensor::activation::threshold;

/// Threshold layer: returns `x` where `x > threshold`, and `value` otherwise.
///
/// Should be created with [ThresholdConfig](ThresholdConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Threshold {
    /// The value to threshold at.
    pub threshold: f64,
    /// The value to replace with where the input is at or below the threshold.
    pub value: f64,
}

/// Configuration to create a [Threshold](Threshold) layer using the [init function](ThresholdConfig::init).
#[derive(Config, Debug)]
pub struct ThresholdConfig {
    /// The value to threshold at.
    pub threshold: f64,
    /// The value to replace with where the input is at or below the threshold.
    pub value: f64,
}

impl ThresholdConfig {
    /// Initialize a new [Threshold](Threshold) layer.
    pub fn init(&self) -> Threshold {
        Threshold {
            threshold: self.threshold,
            value: self.value,
        }
    }
}

impl ModuleDisplay for Threshold {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("threshold", &self.threshold)
            .add("value", &self.value)
            .optional()
    }
}

impl Threshold {
    /// Forward pass for the Threshold layer.
    ///
    /// See [threshold](burn::tensor::activation::threshold) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        threshold(input, self.threshold, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let config = ThresholdConfig::new(1.0, 20.0).init();
        assert_eq!(
            alloc::format!("{config}"),
            "Threshold {threshold: 1, value: 20}"
        );
    }
}
