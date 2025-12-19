use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::adaptive_avg_pool3d;

/// Configuration to create a [3D adaptive avg pooling](AdaptiveAvgPool3d) layer using the [init function](AdaptiveAvgPool3dConfig::init).
#[derive(Config, Debug)]
pub struct AdaptiveAvgPool3dConfig {
    /// The size of the output.
    pub output_size: [usize; 3],
}

/// Applies a 3D adaptive avg pooling over input tensors.
///
/// Should be created with [AdaptiveAvgPool3dConfig].
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct AdaptiveAvgPool3d {
    /// The size of the output.
    pub output_size: [usize; 3],
}

impl ModuleDisplay for AdaptiveAvgPool3d {
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

impl AdaptiveAvgPool3dConfig {
    /// Initialize a new [adaptive avg pool 3d](AdaptiveAvgPool3d) module.
    pub fn init(&self) -> AdaptiveAvgPool3d {
        AdaptiveAvgPool3d {
            output_size: self.output_size,
        }
    }
}

impl AdaptiveAvgPool3d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [adaptive_avg_pool3d](burn::tensor::module::adaptive_avg_pool3d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, depth_in, height_in, width_in]`
    /// - output: `[batch_size, channels, depth_out, height_out, width_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        adaptive_avg_pool3d(input, self.output_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let config = AdaptiveAvgPool3dConfig::new([3, 3, 3]);
        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "AdaptiveAvgPool3d {output_size: [3, 3, 3]}"
        );
    }
}
