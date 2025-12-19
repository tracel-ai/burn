use crate::conv::checks::check_same_padding_support;
use burn_core as burn;

use crate::PaddingConfig3d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::max_pool3d;

/// Configuration to create a [3D max pooling](MaxPool3d) layer using the [init function](MaxPool3dConfig::init).
#[derive(Debug, Config)]
pub struct MaxPool3dConfig {
    /// The size of the kernel.
    pub kernel_size: [usize; 3],
    /// The strides.
    #[config(default = "kernel_size")]
    pub strides: [usize; 3],
    /// The padding configuration.
    ///
    /// ### Warning
    /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
    /// size is not supported as it will not produce the same output size.
    #[config(default = "PaddingConfig3d::Valid")]
    pub padding: PaddingConfig3d,
    /// The dilation.
    #[config(default = "[1, 1, 1]")]
    pub dilation: [usize; 3],
    /// If true, use ceiling instead of floor for output size calculation.
    #[config(default = "false")]
    pub ceil_mode: bool,
}

/// Applies a 3D max pooling over input tensors.
///
/// Should be created with [MaxPool3dConfig](MaxPool3dConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct MaxPool3d {
    /// The strides.
    pub stride: [usize; 3],
    /// The size of the kernel.
    pub kernel_size: [usize; 3],
    /// The padding configuration.
    pub padding: Ignored<PaddingConfig3d>,
    /// The dilation.
    pub dilation: [usize; 3],
    /// If true, use ceiling instead of floor for output size calculation.
    pub ceil_mode: bool,
}

impl ModuleDisplay for MaxPool3d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("kernel_size", &alloc::format!("{:?}", &self.kernel_size))
            .add("stride", &alloc::format!("{:?}", &self.stride))
            .add("padding", &self.padding)
            .add("dilation", &alloc::format!("{:?}", &self.dilation))
            .add("ceil_mode", &self.ceil_mode)
            .optional()
    }
}

impl MaxPool3dConfig {
    /// Initialize a new [max pool 3d](MaxPool3d) module.
    pub fn init(&self) -> MaxPool3d {
        if self.padding == PaddingConfig3d::Same {
            check_same_padding_support(&self.kernel_size);
        }
        MaxPool3d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            dilation: self.dilation,
            ceil_mode: self.ceil_mode,
        }
    }
}

impl MaxPool3d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [max_pool3d](burn::tensor::module::max_pool3d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, depth_in, height_in, width_in]`
    /// - output: `[batch_size, channels, depth_out, height_out, width_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_batch_size, _channels_in, depth_in, height_in, width_in] = input.dims();
        let padding = self.padding.calculate_padding_3d(
            depth_in,
            height_in,
            width_in,
            &self.kernel_size,
            &self.stride,
        );

        max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            padding,
            self.dilation,
            self.ceil_mode,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    #[should_panic = "Same padding with an even kernel size is not supported"]
    fn same_with_even_kernel_is_invalid() {
        let config = MaxPool3dConfig::new([2, 2, 2]).with_padding(PaddingConfig3d::Same);
        let _ = config.init();
    }

    #[test]
    fn display() {
        let config = MaxPool3dConfig::new([3, 3, 3]);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "MaxPool3d {kernel_size: [3, 3, 3], stride: [3, 3, 3], padding: Valid, dilation: [1, 1, 1], ceil_mode: false}"
        );
    }

    #[rstest]
    #[case([2, 2, 2])]
    #[case([1, 2, 3])]
    fn default_strides_match_kernel_size(#[case] kernel_size: [usize; 3]) {
        let config = MaxPool3dConfig::new(kernel_size);

        assert_eq!(
            config.strides, kernel_size,
            "Expected strides ({:?}) to match kernel size ({:?}) in default MaxPool3dConfig::new constructor",
            config.strides, config.kernel_size
        );
    }
}
