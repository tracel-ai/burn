use burn_core as burn;

use crate::PaddingConfig3d;
use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::module::max_pool3d;
use burn::tensor::ops::MaxPoolOptions;

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
    /// `Same` padding may be asymmetric (even kernels, or stride > 1); the input is then pre-padded before pooling.
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
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct MaxPool3d {
    /// The strides.
    pub stride: [usize; 3],
    /// The size of the kernel.
    pub kernel_size: [usize; 3],
    /// The padding configuration.
    ///
    /// `Same` padding may be asymmetric (even kernels, or stride > 1); the input is then pre-padded before pooling.
    #[module(skip)]
    pub padding: PaddingConfig3d,
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
            .add("kernel_size", &alloc::format!("{:?}", self.kernel_size))
            .add("stride", &alloc::format!("{:?}", self.stride))
            .add_debug_attribute("padding", &self.padding)
            .add("dilation", &alloc::format!("{:?}", self.dilation))
            .add("ceil_mode", &self.ceil_mode)
            .optional()
    }
}

impl MaxPool3dConfig {
    /// Initialize a new [max pool 3d](MaxPool3d) module.
    pub fn init(&self) -> MaxPool3d {
        MaxPool3d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
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
    pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
        let [_batch_size, _channels_in, depth_in, height_in, width_in] = input.dims();
        let (padding_depth, padding_height, padding_width) =
            self.padding.calculate_padding_3d_pairs(
                depth_in,
                height_in,
                width_in,
                &self.kernel_size,
                &self.stride,
            );

        max_pool3d(
            input,
            MaxPoolOptions::new(self.kernel_size)
                .with_stride(self.stride)
                .with_padding_pairs([padding_depth, padding_height, padding_width])
                .with_dilation(self.dilation)
                .with_ceil_mode(self.ceil_mode),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn same_with_even_kernel_uses_asymmetric_padding() {
        let device = Default::default();
        let config = MaxPool3dConfig::new([2, 2, 2])
            .with_strides([1, 1, 1])
            .with_padding(PaddingConfig3d::Same);
        let pool = config.init();

        // Input: [batch=1, channels=2, depth=5, height=5, width=5]
        let input = Tensor::<5>::ones([1, 2, 5, 5, 5], &device);
        let output = pool.forward(input);

        // Same padding should preserve spatial dimensions
        assert_eq!(output.dims(), [1, 2, 5, 5, 5]);
    }

    #[test]
    fn display() {
        let config = MaxPool3dConfig::new([3, 3, 3]);
        let pool = config.init();

        assert_eq!(
            alloc::format!("{pool}"),
            "MaxPool3d {kernel_size: [3, 3, 3], stride: [3, 3, 3], padding: Valid, dilation: [1, 1, 1], ceil_mode: false}"
        );
    }

    #[rstest]
    #[case([1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], false, [1, 1, 4, 4, 4])]
    #[case([2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], false, [1, 1, 2, 2, 2])]
    #[case([3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], false, [1, 1, 4, 4, 4])]
    fn symmetric_explicit_padding_forward(
        #[case] kernel_size: [usize; 3],
        #[case] stride: [usize; 3],
        #[case] padding: [usize; 3],
        #[case] dilation: [usize; 3],
        #[case] ceil_mode: bool,
        #[case] expected_dims: [usize; 5],
    ) {
        let device = Default::default();
        let pool = MaxPool3dConfig::new(kernel_size)
            .with_strides(stride)
            .with_padding(PaddingConfig3d::Explicit(
                padding[0], padding[1], padding[2],
            ))
            .with_dilation(dilation)
            .with_ceil_mode(ceil_mode)
            .init();

        let input = Tensor::<5>::ones([1, 1, 4, 4, 4], &device);
        let output = pool.forward(input);

        assert_eq!(output.dims(), expected_dims);
    }

    #[rstest]
    #[case([2, 2, 2], [1, 1, 1], [1, 1, 4, 4, 4])]
    #[case([2, 2, 2], [2, 2, 2], [1, 1, 2, 2, 2])]
    fn asymmetric_padding_forward(
        #[case] kernel_size: [usize; 3],
        #[case] stride: [usize; 3],
        #[case] expected_dims: [usize; 5],
    ) {
        let device = Default::default();
        let pool = MaxPool3dConfig::new(kernel_size)
            .with_strides(stride)
            .with_padding(PaddingConfig3d::Same)
            .init();

        let input = Tensor::<5>::ones([1, 1, 4, 4, 4], &device);
        let output = pool.forward(input);

        assert_eq!(output.dims(), expected_dims);
    }

    #[test]
    fn same_padding_all_negative_no_inf_leak() {
        let device = crate::test_device();
        let input = Tensor::<5>::from_data(
            [[[
                [[-10.0, -20.0], [-30.0, -40.0]],
                [[-50.0, -60.0], [-70.0, -80.0]],
            ]]],
            &device,
        );

        let pool = MaxPool3dConfig::new([2, 2, 2])
            .with_strides([1, 1, 1])
            .with_padding(PaddingConfig3d::Same)
            .init();

        let output = pool.forward(input);
        let expected = Tensor::<5>::from_data(
            [[[
                [[-10.0, -20.0], [-30.0, -40.0]],
                [[-50.0, -60.0], [-70.0, -80.0]],
            ]]],
            &device,
        );
        output.to_data().assert_eq(&expected.to_data(), true);

        // Verify that no output element is -inf or inf
        let output_data = output.into_data();
        let output_slice = output_data.as_slice::<f32>().unwrap();
        for &val in output_slice {
            assert!(
                val.is_finite(),
                "Output value {val} is not finite (leaked -inf)"
            );
            assert!(val < 0.0, "Output value {val} should be negative");
        }
    }
}
