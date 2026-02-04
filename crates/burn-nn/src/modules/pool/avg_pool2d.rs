use burn_core as burn;

use crate::PaddingConfig2d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;

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
    /// Supports symmetric and asymmetric padding. `Same` padding with even kernel sizes
    /// will automatically use asymmetric padding to preserve input dimensions.
    #[config(default = "PaddingConfig2d::Valid")]
    pub padding: PaddingConfig2d,
    /// If the padding is counted in the denominator when computing the average.
    #[config(default = "true")]
    pub count_include_pad: bool,
    /// If true, use ceiling instead of floor for output size calculation.
    #[config(default = "false")]
    pub ceil_mode: bool,
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
    /// If true, use ceiling instead of floor for output size calculation.
    pub ceil_mode: bool,
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
            .add("ceil_mode", &self.ceil_mode)
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
            ceil_mode: self.ceil_mode,
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

        // Calculate padding as pairs - handles Same, Valid, and Explicit uniformly
        let ((top, bottom), (left, right)) = self.padding.calculate_padding_2d_pairs(
            height_in,
            width_in,
            &self.kernel_size,
            &self.stride,
        );

        // TODO: Move asymmetric padding to functional level via PoolOptions
        // See: https://github.com/tracel-ai/burn/issues/4362
        // Handle asymmetric padding by applying explicit pad operation first
        if top != bottom || left != right {
            // Burn's pad takes (left, right, top, bottom) for the last two dimensions
            let padded = input.pad((left, right, top, bottom), PadMode::Constant(0.0));
            // Use zero padding for the pool operation since we already padded
            avg_pool2d(
                padded,
                self.kernel_size,
                self.stride,
                [0, 0],
                self.count_include_pad,
                self.ceil_mode,
            )
        } else {
            // Symmetric padding
            avg_pool2d(
                input,
                self.kernel_size,
                self.stride,
                [top, left],
                self.count_include_pad,
                self.ceil_mode,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use rstest::rstest;

    #[test]
    fn same_with_even_kernel_uses_asymmetric_padding() {
        let device = Default::default();
        let config = AvgPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Same);
        let pool = config.init();

        // Input: [batch=1, channels=2, height=5, width=5]
        let input = Tensor::<TestBackend, 4>::ones([1, 2, 5, 5], &device);
        let output = pool.forward(input);

        // Same padding should preserve spatial dimensions
        assert_eq!(output.dims(), [1, 2, 5, 5]);
    }

    #[test]
    fn display() {
        let config = AvgPool2dConfig::new([3, 3]);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "AvgPool2d {kernel_size: [3, 3], stride: [3, 3], padding: Valid, count_include_pad: true, ceil_mode: false}"
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

    #[test]
    fn asymmetric_padding_forward() {
        let device = Default::default();
        // Create avg pool with asymmetric padding: top=1, left=2, bottom=3, right=4
        let config = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 2, 3, 4));
        let pool = config.init();

        // Input: [batch=1, channels=2, height=4, width=5]
        let input = Tensor::<TestBackend, 4>::ones([1, 2, 4, 5], &device);
        let output = pool.forward(input);

        // Height: 4 + 1 + 3 = 8, output = (8 - 3) / 1 + 1 = 6
        // Width: 5 + 2 + 4 = 11, output = (11 - 3) / 1 + 1 = 9
        assert_eq!(output.dims(), [1, 2, 6, 9]);
    }

    #[test]
    fn symmetric_explicit_padding_forward() {
        let device = Default::default();
        // Create avg pool with symmetric explicit padding: top=2, left=2, bottom=2, right=2
        let config = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2));
        let pool = config.init();

        // Input: [batch=1, channels=2, height=4, width=5]
        let input = Tensor::<TestBackend, 4>::ones([1, 2, 4, 5], &device);
        let output = pool.forward(input);

        // Height: 4 + 2 + 2 = 8, output = (8 - 3) / 1 + 1 = 6
        // Width: 5 + 2 + 2 = 9, output = (9 - 3) / 1 + 1 = 7
        assert_eq!(output.dims(), [1, 2, 6, 7]);
    }
}
