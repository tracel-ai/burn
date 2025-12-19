use crate::conv::checks::check_same_padding_support;
use burn_core as burn;

use crate::PaddingConfig3d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use burn::tensor::module::avg_pool3d;

/// Configuration to create a [3D avg pooling](AvgPool3d) layer using the [init function](AvgPool3dConfig::init).
#[derive(Config, Debug)]
pub struct AvgPool3dConfig {
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
    /// If the padding is counted in the denominator when computing the average.
    #[config(default = "true")]
    pub count_include_pad: bool,
    /// If true, use ceiling instead of floor for output size calculation.
    #[config(default = "false")]
    pub ceil_mode: bool,
}

/// Applies a 3D avg pooling over input tensors.
///
/// Should be created with [AvgPool3dConfig](AvgPool3dConfig).
///
/// # Remarks
///
/// The zero-padding values will be included in the calculation
/// of the average. This means that the zeros are counted as
/// legitimate values, and they contribute to the denominator
/// when calculating the average. This is equivalent to
/// `torch.nn.AvgPool3d` with `count_include_pad=True`.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct AvgPool3d {
    /// Stride of the pooling.
    pub stride: [usize; 3],
    /// Size of the kernel.
    pub kernel_size: [usize; 3],
    /// Padding configuration.
    pub padding: Ignored<PaddingConfig3d>,
    /// If the padding is counted in the denominator when computing the average.
    pub count_include_pad: bool,
    /// If true, use ceiling instead of floor for output size calculation.
    pub ceil_mode: bool,
}

impl ModuleDisplay for AvgPool3d {
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
            .add("ceil_mode", &self.ceil_mode)
            .optional()
    }
}

impl AvgPool3dConfig {
    /// Initialize a new [avg pool 3d](AvgPool3d) module.
    pub fn init(&self) -> AvgPool3d {
        if self.padding == PaddingConfig3d::Same {
            check_same_padding_support(&self.kernel_size);
        }
        AvgPool3d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            count_include_pad: self.count_include_pad,
            ceil_mode: self.ceil_mode,
        }
    }
}

impl AvgPool3d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [avg_pool3d](burn::tensor::module::avg_pool3d) for more information.
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

        avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            padding,
            self.count_include_pad,
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
        let config = AvgPool3dConfig::new([2, 2, 2]).with_padding(PaddingConfig3d::Same);
        let _ = config.init();
    }

    #[test]
    fn display() {
        let config = AvgPool3dConfig::new([3, 3, 3]);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "AvgPool3d {kernel_size: [3, 3, 3], stride: [3, 3, 3], padding: Valid, count_include_pad: true, ceil_mode: false}"
        );
    }

    #[rstest]
    #[case([2, 2, 2])]
    #[case([1, 2, 3])]
    fn default_strides_match_kernel_size(#[case] kernel_size: [usize; 3]) {
        let config = AvgPool3dConfig::new(kernel_size);

        assert_eq!(
            config.strides, kernel_size,
            "Expected strides ({:?}) to match kernel size ({:?}) in default AvgPool3dConfig::new constructor",
            config.strides, config.kernel_size
        );
    }
}
