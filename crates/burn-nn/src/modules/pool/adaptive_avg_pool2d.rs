use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::adaptive_avg_pool2d;

/// Configuration to create a [2D adaptive avg pooling](AdaptiveAvgPool2d) layer using the [init function](AdaptiveAvgPool2dConfig::init).
#[derive(Config, Debug)]
pub struct AdaptiveAvgPool2dConfig {
    /// The size of the output.
    pub output_size: [usize; 2],
}

/// Applies a 2D adaptive avg pooling over input tensors.
///
/// Should be created with [AdaptiveAvgPool2dConfig].
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct AdaptiveAvgPool2d {
    /// The size of the output.
    pub output_size: [usize; 2],
}

impl ModuleDisplay for AdaptiveAvgPool2d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let output_size = alloc::format!("{:?}", self.output_size);

        content.add("output_size", &output_size).optional()
    }
}

impl AdaptiveAvgPool2dConfig {
    /// Initialize a new [adaptive avg pool 2d](AdaptiveAvgPool2d) module.
    pub fn init(&self) -> AdaptiveAvgPool2d {
        AdaptiveAvgPool2d {
            output_size: self.output_size,
        }
    }
}

impl AdaptiveAvgPool2d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [adaptive_avg_pool2d](burn::tensor::module::adaptive_avg_pool2d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, height_in, width_in]`
    /// - output: `[batch_size, channels, height_out, width_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        adaptive_avg_pool2d(input, self.output_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let config = AdaptiveAvgPool2dConfig::new([3, 3]);
        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "AdaptiveAvgPool2d {output_size: [3, 3]}"
        );
    }
}
