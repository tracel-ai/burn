use crate::conv::checks::check_same_padding_support;
use burn_core as burn;

use crate::PaddingConfig2d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::avg_pool2d;

/// Configuration to create a [2D avg pooling](AvgPool2d) layer using the [init function](AvgPool2dConfig::init).
#[derive(Config, Debug)]
pub struct AvgPool2dConfig {
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
    /// If the padding is counted in the denominator when computing the average.
    #[config(default = "true")]
    pub count_include_pad: bool,
}

/// Applies a 2D avg pooling over input tensors.
///
/// Should be created with [AvgPool2dConfig](AvgPool2dConfig).
///
/// # Remarks
///
/// The zero-padding values will be included in the calculation
/// of the average. This means that the zeros are counted as
/// legitimate values, and they contribute to the denominator
/// when calculating the average. This is equivalent to
/// `torch.nn.AvgPool2d` with `count_include_pad=True`.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct AvgPool2d {
    /// Stride of the pooling.
    pub stride: [usize; 2],
    /// Size of the kernel.
    pub kernel_size: [usize; 2],
    /// Padding configuration.
    pub padding: Ignored<PaddingConfig2d>,
    /// If the padding is counted in the denominator when computing the average.
    pub count_include_pad: bool,
}

impl ModuleDisplay for AvgPool2d {
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
            .add("count_include_pad", &self.count_include_pad)
            .optional()
    }
}

impl AvgPool2dConfig {
    /// Initialize a new [avg pool 2d](AvgPool2d) module.
    pub fn init(&self) -> AvgPool2d {
        if self.padding == PaddingConfig2d::Same {
            check_same_padding_support(&self.kernel_size);
        }
        AvgPool2d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            count_include_pad: self.count_include_pad,
        }
    }
}

impl AvgPool2d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [avg_pool2d](burn::tensor::module::avg_pool2d) for more information.
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

        avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            padding,
            self.count_include_pad,
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
        let config = AvgPool2dConfig::new([2, 2]).with_padding(PaddingConfig2d::Same);
        let _ = config.init();
    }

    #[test]
    fn display() {
        let config = AvgPool2dConfig::new([3, 3]);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "AvgPool2d {kernel_size: [3, 3], stride: [3, 3], padding: Valid, count_include_pad: true}"
        );
    }

    #[rstest]
    #[case([2, 2])]
    #[case([1, 2])]
    fn default_strides_match_kernel_size(#[case] kernel_size: [usize; 2]) {
        let config = AvgPool2dConfig::new(kernel_size);

        assert_eq!(
            config.strides, kernel_size,
            "Expected strides ({:?}) to match kernel size ({:?}) in default AvgPool2dConfig::new constructor",
            config.strides, config.kernel_size
        );
    }
}
