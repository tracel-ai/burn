use crate as burn;

use crate::config::Config;
use crate::module::{Content, DisplaySettings, ModuleDisplay};
use crate::module::{Ignored, Module};
use crate::nn::PaddingConfig2d;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use crate::tensor::module::avg_pool2d;

/// Configuration to create a [2D avg pooling](AvgPool2d) layer using the [init function](AvgPool2dConfig::init).
#[derive(Config, Debug)]
pub struct AvgPool2dConfig {
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The strides.
    #[config(default = "[1, 1]")]
    pub strides: [usize; 2],
    /// The padding configuration.
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
///
/// TODO: Add support for `count_include_pad=False`, see
/// [Issue 636](https://github.com/tracel-ai/burn/issues/636)
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
    /// See [avg_pool2d](crate::tensor::module::avg_pool2d) for more information.
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

    #[test]
    fn display() {
        let config = AvgPool2dConfig::new([3, 3]);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{}", layer),
            "AvgPool2d {kernel_size: [3, 3], stride: [1, 1], padding: Valid, count_include_pad: true}"
        );
    }
}
