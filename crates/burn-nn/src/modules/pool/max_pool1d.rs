use burn_core as burn;

use crate::PaddingConfig1d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;

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
    /// Supports symmetric and asymmetric padding. `Same` padding with even kernel sizes
    /// will automatically use asymmetric padding to preserve input dimensions.
    #[config(default = "PaddingConfig1d::Valid")]
    pub padding: PaddingConfig1d,
    /// The dilation.
    #[config(default = "1")]
    pub dilation: usize,
    /// If true, use ceiling instead of floor for output size calculation.
    #[config(default = "false")]
    pub ceil_mode: bool,
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
    /// If true, use ceiling instead of floor for output size calculation.
    pub ceil_mode: bool,
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
            .add("ceil_mode", &self.ceil_mode)
            .optional()
    }
}

impl MaxPool1dConfig {
    /// Initialize a new [max pool 1d](MaxPool1d) module.
    pub fn init(&self) -> MaxPool1d {
        MaxPool1d {
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            dilation: self.dilation,
            ceil_mode: self.ceil_mode,
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

        // Calculate padding as pair - handles Same, Valid, and Explicit uniformly
        let (left, right) =
            self.padding
                .calculate_padding_1d_pair(length, self.kernel_size, self.stride);

        // TODO: Move asymmetric padding to functional level via PoolOptions
        // See: https://github.com/tracel-ai/burn/issues/4362
        // Handle asymmetric padding by applying explicit pad operation first
        if left != right {
            // Burn's pad takes (left, right, top, bottom) for the last two dimensions
            // For 1D (NCL format), we only pad L (last dim), so top/bottom = 0
            // Use -inf for max pooling so padded values don't affect the max
            let padded = input.pad((left, right, 0, 0), PadMode::Constant(f32::NEG_INFINITY));
            // Use zero padding for the pool operation since we already padded
            max_pool1d(
                padded,
                self.kernel_size,
                self.stride,
                0,
                self.dilation,
                self.ceil_mode,
            )
        } else {
            // Symmetric padding
            max_pool1d(
                input,
                self.kernel_size,
                self.stride,
                left,
                self.dilation,
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
        let config = MaxPool1dConfig::new(2)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Same);
        let pool = config.init();

        // Input: [batch=1, channels=2, length=5]
        let input = Tensor::<TestBackend, 3>::ones([1, 2, 5], &device);
        let output = pool.forward(input);

        // Same padding should preserve spatial dimensions
        assert_eq!(output.dims(), [1, 2, 5]);
    }

    #[test]
    fn display() {
        let config = MaxPool1dConfig::new(3);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "MaxPool1d {kernel_size: 3, stride: 3, padding: Valid, dilation: 1, ceil_mode: false}"
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

    #[test]
    fn asymmetric_padding_forward() {
        let device = Default::default();
        // Create max pool with asymmetric padding: left=1, right=2
        let config = MaxPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 2));
        let pool = config.init();

        // Input: [batch=1, channels=2, length=4]
        let input = Tensor::<TestBackend, 3>::ones([1, 2, 4], &device);
        let output = pool.forward(input);

        // With asymmetric padding (1, 2), input length 4 becomes 4+1+2=7
        // Output length = (7 - 3) / 1 + 1 = 5
        assert_eq!(output.dims(), [1, 2, 5]);
    }

    #[test]
    fn symmetric_explicit_padding_forward() {
        let device = Default::default();
        // Create max pool with symmetric explicit padding: left=2, right=2
        let config = MaxPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2));
        let pool = config.init();

        // Input: [batch=1, channels=2, length=4]
        let input = Tensor::<TestBackend, 3>::ones([1, 2, 4], &device);
        let output = pool.forward(input);

        // With symmetric padding (2, 2), input length 4 becomes 4+2+2=8
        // Output length = (8 - 3) / 1 + 1 = 6
        assert_eq!(output.dims(), [1, 2, 6]);
    }
}
