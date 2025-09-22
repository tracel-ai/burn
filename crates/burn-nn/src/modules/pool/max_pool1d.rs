use crate::conv::checks::check_same_padding_support;
use burn_core as burn;

use crate::PaddingConfig1d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::max_pool1d;

/// Configuration to create a [1D max pooling](MaxPool1d) layer using the [init function](MaxPool1dConfig::init).
#[derive(Config, Debug)]
pub struct MaxPool1dConfig {
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
    /// The dilation.
    #[config(default = "1")]
    pub dilation: usize,
}

/// Applies a 1D max pooling over input tensors.
///
/// Should be created with [MaxPool1dConfig](MaxPool1dConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct MaxPool1d {
    /// The stride.
    pub stride: usize,
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The padding configuration.
    pub padding: Ignored<PaddingConfig1d>,
    /// The dilation.
    pub dilation: usize,
}

impl ModuleDisplay for MaxPool1d {
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
            .add("dilation", &self.dilation)
            .optional()
    }
}

impl MaxPool1dConfig {
    /// Initialize a new [max pool 1d](MaxPool1d) module.
    pub fn init(&self) -> MaxPool1d {
        if self.padding == PaddingConfig1d::Same {
            check_same_padding_support(&[self.kernel_size]);
        }
        MaxPool1d {
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            dilation: self.dilation,
        }
    }
}

impl MaxPool1d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [max_pool1d](burn::tensor::module::max_pool1d) for more information.
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

        max_pool1d(input, self.kernel_size, self.stride, padding, self.dilation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    #[should_panic = "Same padding with an even kernel size is not supported"]
    fn same_with_even_kernel_is_invalid() {
        let config = MaxPool1dConfig::new(2).with_padding(PaddingConfig1d::Same);
        let _ = config.init();
    }

    #[test]
    fn display() {
        let config = MaxPool1dConfig::new(3);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "MaxPool1d {kernel_size: 3, stride: 3, padding: Valid, dilation: 1}"
        );
    }

    #[rstest]
    #[case(1)]
    #[case(2)]
    fn default_strides_match_kernel_size(#[case] kernel_size: usize) {
        let config = MaxPool1dConfig::new(kernel_size);

        assert_eq!(
            config.stride, kernel_size,
            "Expected stride ({:?}) to match kernel size ({:?}) in default MaxPool1dConfig::new constructor",
            config.stride, config.kernel_size
        );
    }
}
