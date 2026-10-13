use burn_core as burn;

use crate::PaddingConfig3d;
use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::module::avg_pool3d;
use burn::tensor::ops::AvgPoolOptions;

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
    /// `Same` padding may be asymmetric (even kernels, or stride > 1); the input is then pre-padded before pooling.
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
/// By default (`count_include_pad = true`), zero-padding values are included in the calculation.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct AvgPool3d {
    /// Stride of the pooling.
    pub stride: [usize; 3],
    /// Size of the kernel.
    pub kernel_size: [usize; 3],
    /// Padding configuration.
    ///
    /// `Same` padding may be asymmetric (even kernels, or stride > 1); the input is then pre-padded before pooling.
    #[module(skip)]
    pub padding: PaddingConfig3d,
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
            .add("kernel_size", &alloc::format!("{:?}", self.kernel_size))
            .add("stride", &alloc::format!("{:?}", self.stride))
            .add_debug_attribute("padding", &self.padding)
            .add("count_include_pad", &self.count_include_pad)
            .add("ceil_mode", &self.ceil_mode)
            .optional()
    }
}

impl AvgPool3dConfig {
    /// Initialize a new [avg pool 3d](AvgPool3d) module.
    pub fn init(&self) -> AvgPool3d {
        AvgPool3d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
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

        avg_pool3d(
            input,
            AvgPoolOptions::new(self.kernel_size)
                .with_stride(self.stride)
                .with_padding_pairs([padding_depth, padding_height, padding_width])
                .with_count_include_pad(self.count_include_pad)
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
        let config = AvgPool3dConfig::new([2, 2, 2])
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
        let config = AvgPool3dConfig::new([3, 3, 3]);
        let pool = config.init();

        assert_eq!(
            alloc::format!("{pool}"),
            "AvgPool3d {kernel_size: [3, 3, 3], stride: [3, 3, 3], padding: Valid, count_include_pad: true, ceil_mode: false}"
        );
    }

    #[rstest]
    #[case([1, 1, 1], [1, 1, 1], [0, 0, 0], false, [1, 1, 4, 4, 4])]
    #[case([2, 2, 2], [2, 2, 2], [0, 0, 0], false, [1, 1, 2, 2, 2])]
    #[case([3, 3, 3], [1, 1, 1], [1, 1, 1], false, [1, 1, 4, 4, 4])]
    fn symmetric_explicit_padding_forward(
        #[case] kernel_size: [usize; 3],
        #[case] stride: [usize; 3],
        #[case] padding: [usize; 3],
        #[case] ceil_mode: bool,
        #[case] expected_dims: [usize; 5],
    ) {
        let device = Default::default();
        let pool = AvgPool3dConfig::new(kernel_size)
            .with_strides(stride)
            .with_padding(PaddingConfig3d::Explicit(
                padding[0], padding[1], padding[2],
            ))
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
        let pool = AvgPool3dConfig::new(kernel_size)
            .with_strides(stride)
            .with_padding(PaddingConfig3d::Same)
            .init();

        let input = Tensor::<5>::ones([1, 1, 4, 4, 4], &device);
        let output = pool.forward(input);

        assert_eq!(output.dims(), expected_dims);
    }

    #[test]
    fn asymmetric_padding_excludes_pad_from_average() {
        let device = crate::test_device();
        let config = AvgPool3dConfig::new([2, 2, 2])
            .with_strides([1, 1, 1])
            .with_padding(PaddingConfig3d::Same)
            .with_count_include_pad(false);
        let pool = config.init();

        let input = Tensor::<5>::ones([1, 1, 3, 3, 3], &device);
        let output = pool.forward(input);

        // Every window contains at least one 1.0 from the input, and since all input values are 1.0
        // and padding is excluded from the average, the result must be 1.0 everywhere.
        let expected = Tensor::<5>::ones([1, 1, 3, 3, 3], &device);
        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn same_padding_even_kernel_arange_counts() {
        let device = crate::test_device();
        let input = Tensor::<1, burn::tensor::Int>::arange(0..8, &device)
            .float()
            .reshape([1, 1, 2, 2, 2]);

        // count_include_pad = true
        let pool_inc = AvgPool3dConfig::new([2, 2, 2])
            .with_strides([1, 1, 1])
            .with_padding(PaddingConfig3d::Same)
            .with_count_include_pad(true)
            .init();
        let out_inc = pool_inc.forward(input.clone());
        let expected_inc = Tensor::<5>::from_data(
            [[[[[3.5, 2.0], [2.25, 1.25]], [[2.75, 1.5], [1.625, 0.875]]]]],
            &device,
        );
        out_inc.to_data().assert_eq(&expected_inc.to_data(), true);

        // count_include_pad = false
        let pool_exc = AvgPool3dConfig::new([2, 2, 2])
            .with_strides([1, 1, 1])
            .with_padding(PaddingConfig3d::Same)
            .with_count_include_pad(false)
            .init();
        let out_exc = pool_exc.forward(input);
        let expected_exc = Tensor::<5>::from_data(
            [[[[[3.5, 4.0], [4.5, 5.0]], [[5.5, 6.0], [6.5, 7.0]]]]],
            &device,
        );
        out_exc.to_data().assert_eq(&expected_exc.to_data(), true);
    }
}
