use crate::conv::checks::check_same_padding_support;
use burn_core as burn;

use crate::PaddingConfig2d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::max_pool2d;

/// Configuration to create a [2D max pooling](MaxPool2d) layer using the [init function](MaxPool2dConfig::init).
#[derive(Debug, Config)]
pub struct MaxPool2dConfig {
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The strides.
    #[config(default = "kernel_size")]
    pub strides: [usize; 2],
    /// The padding configuration.
    ///
    /// ### Warning
    /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
    /// size is not supported as it will not produce the same output size.
    #[config(default = "PaddingConfig2d::Valid")]
    pub padding: PaddingConfig2d,
    /// The dilation.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
}

/// Applies a 2D max pooling over input tensors.
///
/// Should be created with [MaxPool2dConfig](MaxPool2dConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct MaxPool2d {
    /// The strides.
    pub stride: [usize; 2],
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The padding configuration.
    pub padding: Ignored<PaddingConfig2d>,
    /// The dilation.
    pub dilation: [usize; 2],
}

impl ModuleDisplay for MaxPool2d {
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
            .optional()
    }
}

impl MaxPool2dConfig {
    /// Initialize a new [max pool 2d](MaxPool2d) module.
    pub fn init(&self) -> MaxPool2d {
        if self.padding == PaddingConfig2d::Same {
            check_same_padding_support(&self.kernel_size);
        }
        MaxPool2d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            dilation: self.dilation,
        }
    }
}

impl MaxPool2d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [max_pool2d](burn::tensor::module::max_pool2d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, height_in, width_in]`
    /// - output: `[batch_size, channels, height_out, width_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, _channels_in, height_in, width_in] = input.dims();
        let padding =
            self.padding
                .calculate_padding_2d(height_in, width_in, &self.kernel_size, &self.stride);

        max_pool2d(input, self.kernel_size, self.stride, padding, self.dilation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    #[should_panic = "Same padding with an even kernel size is not supported"]
    fn same_with_even_kernel_is_invalid() {
        let config = MaxPool2dConfig::new([2, 2]).with_padding(PaddingConfig2d::Same);
        let _ = config.init();
    }

    #[test]
    fn display() {
        let config = MaxPool2dConfig::new([3, 3]);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "MaxPool2d {kernel_size: [3, 3], stride: [3, 3], padding: Valid, dilation: [1, 1]}"
        );
    }

    #[rstest]
    #[case([2, 2])]
    #[case([1, 2])]
    fn default_strides_match_kernel_size(#[case] kernel_size: [usize; 2]) {
        let config = MaxPool2dConfig::new(kernel_size);

        assert_eq!(
            config.strides, kernel_size,
            "Expected strides ({:?}) to match kernel size ({:?}) in default MaxPool2dConfig::new constructor",
            config.strides, config.kernel_size
        );
    }
}
