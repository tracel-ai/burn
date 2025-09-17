use crate::conv::checks::check_same_padding_support;
use burn_core as burn;

use crate::PaddingConfig1d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::avg_pool1d;

/// Configuration to create a [1D avg pooling](AvgPool1d) layer using the [init function](AvgPool1dConfig::init).
#[derive(Config, Debug)]
pub struct AvgPool1dConfig {
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The stride.
    #[config(default = "kernel_size")]
    pub stride: usize,
    /// The padding configuration.
    ///
    /// ### Warning
    /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
    /// size is not supported as it will not produce the same output size.
    #[config(default = "PaddingConfig1d::Valid")]
    pub padding: PaddingConfig1d,
    /// If the padding is counted in the denominator when computing the average.
    #[config(default = "true")]
    pub count_include_pad: bool,
}

/// Applies a 1D avg pooling over input tensors.
///
/// Should be created with [AvgPool1dConfig](AvgPool1dConfig).
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
pub struct AvgPool1d {
    /// The stride.
    pub stride: usize,
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The padding configuration.
    pub padding: Ignored<PaddingConfig1d>,
    /// If the padding is counted in the denominator when computing the average.
    pub count_include_pad: bool,
}

impl ModuleDisplay for AvgPool1d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("kernel_size", &self.kernel_size)
            .add("stride", &self.stride)
            .add("padding", &self.padding)
            .add("count_include_pad", &self.count_include_pad)
            .optional()
    }
}

impl AvgPool1dConfig {
    /// Initialize a new [avg pool 1d](AvgPool1d) module.
    pub fn init(&self) -> AvgPool1d {
        if self.padding == PaddingConfig1d::Same {
            check_same_padding_support(&[self.kernel_size]);
        }
        AvgPool1d {
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            count_include_pad: self.count_include_pad,
        }
    }
}

impl AvgPool1d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [avg_pool1d](burn::tensor::module::avg_pool1d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, length_in]`
    /// - output: `[batch_size, channels, length_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch_size, _channels, length] = input.dims();
        let padding = self
            .padding
            .calculate_padding_1d(length, self.kernel_size, self.stride);

        avg_pool1d(
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
        let config = AvgPool1dConfig::new(2).with_padding(PaddingConfig1d::Same);
        let _ = config.init();
    }

    #[test]
    fn display() {
        let config = AvgPool1dConfig::new(3);
        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "AvgPool1d {kernel_size: 3, stride: 3, padding: Valid, count_include_pad: true}"
        );
    }

    #[rstest]
    #[case(1)]
    #[case(2)]
    fn default_strides_match_kernel_size(#[case] kernel_size: usize) {
        let config = AvgPool1dConfig::new(kernel_size);

        assert_eq!(
            config.stride, kernel_size,
            "Expected stride ({:?}) to match kernel size ({:?}) in default AvgPool1dConfig::new constructor",
            config.stride, config.kernel_size
        );
    }
}
