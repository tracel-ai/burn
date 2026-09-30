use burn_core as burn;

use crate::PaddingConfig1d;
use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::ops::AvgPoolOptions;

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
    /// Supports symmetric and asymmetric padding. `Same` padding with even kernel sizes
    /// will automatically use asymmetric padding to preserve input dimensions.
    #[config(default = "PaddingConfig1d::Valid")]
    pub padding: PaddingConfig1d,
    /// If the padding is counted in the denominator when computing the average.
    #[config(default = "true")]
    pub count_include_pad: bool,
    /// If true, use ceiling instead of floor for output size calculation.
    #[config(default = "false")]
    pub ceil_mode: bool,
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
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct AvgPool1d {
    /// The stride.
    pub stride: usize,
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The padding configuration.
    #[module(skip)]
    pub padding: PaddingConfig1d,
    /// If the padding is counted in the denominator when computing the average.
    pub count_include_pad: bool,
    /// If true, use ceiling instead of floor for output size calculation.
    pub ceil_mode: bool,
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
            .add_debug_attribute("padding", &self.padding)
            .add("count_include_pad", &self.count_include_pad)
            .add("ceil_mode", &self.ceil_mode)
            .optional()
    }
}

impl AvgPool1dConfig {
    /// Initialize a new [avg pool 1d](AvgPool1d) module.
    pub fn init(&self) -> AvgPool1d {
        AvgPool1d {
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
            count_include_pad: self.count_include_pad,
            ceil_mode: self.ceil_mode,
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
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [_batch_size, _channels, length] = input.dims();
        let padding = self
            .padding
            .calculate_padding_1d_pair(length, self.kernel_size, self.stride);

        avg_pool1d(
            input,
            AvgPoolOptions::new([self.kernel_size])
                .with_stride([self.stride])
                .with_padding_pairs([padding])
                .with_count_include_pad(self.count_include_pad)
                .with_ceil_mode(self.ceil_mode),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData, Tolerance};
    use rstest::rstest;

    #[test]
    fn same_with_even_kernel_uses_asymmetric_padding() {
        let device = Default::default();
        let config = AvgPool1dConfig::new(2)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Same);
        let pool = config.init();

        // Input: [batch=1, channels=2, length=5]
        let input = Tensor::<3>::ones([1, 2, 5], &device);
        let output = pool.forward(input);

        // Same padding should preserve spatial dimensions
        assert_eq!(output.dims(), [1, 2, 5]);
    }

    #[test]
    fn display() {
        let config = AvgPool1dConfig::new(3);
        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "AvgPool1d {kernel_size: 3, stride: 3, padding: Valid, count_include_pad: true, ceil_mode: false}"
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

    #[test]
    fn asymmetric_padding_forward() {
        let device = Default::default();
        // Create avg pool with asymmetric padding: left=1, right=2
        let config = AvgPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 2));
        let pool = config.init();

        // Input: [batch=1, channels=2, length=4]
        let input = Tensor::<3>::ones([1, 2, 4], &device);
        let output = pool.forward(input);

        // With asymmetric padding (1, 2), input length 4 becomes 4+1+2=7
        // Output length = (7 - 3) / 1 + 1 = 5
        assert_eq!(output.dims(), [1, 2, 5]);
    }

    #[test]
    fn asymmetric_padding_mask_broadcasts_without_excluding_input_zeros() {
        let device = Default::default();
        let input = Tensor::from_data(
            [
                [[0.0f32, 2.0, 4.0], [2.0, 0.0, 6.0]],
                [[0.0, 0.0, 8.0], [-2.0, 2.0, 0.0]],
            ],
            &device,
        );
        let pool = AvgPool1dConfig::new(2)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(0, 1))
            .with_count_include_pad(false)
            .init();

        let output = pool.forward(input);

        output.to_data().assert_eq(
            &TensorData::from([[[1.0f32, 4.0], [1.0, 6.0]], [[0.0, 8.0], [0.0, 0.0]]]),
            true,
        );
    }

    #[test]
    fn same_asymmetric_padding_excludes_pad_from_average() {
        let device = Default::default();
        let input = Tensor::from_data([[[1.0f32, 2.0, 3.0, 4.0, 5.0]]], &device);
        let pool = AvgPool1dConfig::new(2)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Same)
            .with_count_include_pad(false)
            .init();

        let output = pool.forward(input);

        output
            .to_data()
            .assert_eq(&TensorData::from([[[1.5f32, 3.5, 5.0]]]), true);
    }

    #[test]
    fn asymmetric_padding_still_counts_pad_when_enabled() {
        let device = Default::default();
        let input = Tensor::from_data([[[1.0f32, 2.0, 3.0, 4.0, 5.0]]], &device);
        let pool = AvgPool1dConfig::new(2)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(0, 1))
            .with_count_include_pad(true)
            .init();

        let output = pool.forward(input);

        output
            .to_data()
            .assert_eq(&TensorData::from([[[1.5f32, 3.5, 2.5]]]), true);
    }

    #[test]
    fn ceil_mode_excludes_asymmetric_pad_and_preserves_gradients() {
        let device = Device::default().autodiff();
        let input = Tensor::from_data([[[2.0f32, 4.0, 6.0, 8.0]]], &device).require_grad();
        let pool = AvgPool1dConfig::new(2)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 0))
            .with_count_include_pad(false)
            .with_ceil_mode(true)
            .init();

        let output = pool.forward(input.clone());
        output
            .clone()
            .to_data()
            .assert_eq(&TensorData::from([[[2.0f32, 5.0, 8.0]]]), true);

        let gradients = output.sum().backward();
        input
            .grad(&gradients)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[[1.0f32, 0.5, 0.5, 1.0]]]),
                Tolerance::default(),
            );
    }

    #[test]
    fn symmetric_explicit_padding_forward() {
        let device = Default::default();
        // Create avg pool with symmetric explicit padding: left=2, right=2
        let config = AvgPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2));
        let pool = config.init();

        // Input: [batch=1, channels=2, length=4]
        let input = Tensor::<3>::ones([1, 2, 4], &device);
        let output = pool.forward(input);

        // With symmetric padding (2, 2), input length 4 becomes 4+2+2=8
        // Output length = (8 - 3) / 1 + 1 = 6
        assert_eq!(output.dims(), [1, 2, 6]);
    }
}
