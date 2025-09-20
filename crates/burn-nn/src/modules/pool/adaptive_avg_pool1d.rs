use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::adaptive_avg_pool1d;

/// Configuration to create a [1D adaptive avg pooling](AdaptiveAvgPool1d) layer using the [init function](AdaptiveAvgPool1dConfig::init).
#[derive(Config, Debug)]
pub struct AdaptiveAvgPool1dConfig {
    /// The size of the output.
    pub output_size: usize,
}

/// Applies a 1D adaptive avg pooling over input tensors.
///
/// Should be created with [AdaptiveAvgPool1dConfig].
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct AdaptiveAvgPool1d {
    /// The size of the output.
    pub output_size: usize,
}

impl ModuleDisplay for AdaptiveAvgPool1d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("output_size", &self.output_size).optional()
    }
}

impl AdaptiveAvgPool1dConfig {
    /// Initialize a new [adaptive avg pool 1d](AdaptiveAvgPool1d) module.
    pub fn init(&self) -> AdaptiveAvgPool1d {
        AdaptiveAvgPool1d {
            output_size: self.output_size,
        }
    }
}

impl AdaptiveAvgPool1d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [adaptive_avg_pool1d](burn::tensor::module::adaptive_avg_pool1d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, length]`
    /// - output: `[batch_size, channels, length_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        adaptive_avg_pool1d(input, self.output_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let config = AdaptiveAvgPool1dConfig::new(3);
        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "AdaptiveAvgPool1d {output_size: 3}"
        );
    }
}
