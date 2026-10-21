use crate::modules::bounds::{self, Windows};
use burn::instrument::{Bounds, compute};
use burn::tensor::module::max_pool_output_size;
use burn_core as burn;

use crate::PaddingConfig2d;
use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::ops::MaxPoolOptions;

use burn::tensor::module::max_pool2d;

/// Configuration to create a [2D max pooling](MaxPool2d) layer using the [init function](MaxPool2dConfig::init).
#[derive(Debug, Config)]
pub struct MaxPool2dConfig {
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The strides.
    #[config(default = "kernel_size")]
    pub strides: [usize; 2],
    /// The padding configuration.
    ///
    /// `Same` padding produces `ceil(input_size / stride)` outputs per spatial dimension.
    /// Padding may be asymmetric, with the extra element added at the end.
    #[config(default = "PaddingConfig2d::Valid")]
    pub padding: PaddingConfig2d,
    /// The dilation.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
    /// If true, use ceiling instead of floor for output size calculation.
    #[config(default = "false")]
    pub ceil_mode: bool,
}

/// Applies a 2D max pooling over input tensors.
///
/// Should be created with [MaxPool2dConfig](MaxPool2dConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct MaxPool2d {
    /// The strides.
    pub stride: [usize; 2],
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The padding configuration.
    #[module(skip)]
    pub padding: PaddingConfig2d,
    /// The dilation.
    pub dilation: [usize; 2],
    /// If true, use ceiling instead of floor for output size calculation.
    pub ceil_mode: bool,
}

impl ModuleDisplay for MaxPool2d {
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

impl MaxPool2dConfig {
    /// Initialize a new [max pool 2d](MaxPool2d) module.
    pub fn init(&self) -> MaxPool2d {
        MaxPool2d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
            dilation: self.dilation,
            ceil_mode: self.ceil_mode,
        }
    }
}

impl MaxPool2d {
    /// The options the pool runs with, its padding resolved against `input`'s size.
    fn options(&self, input: &Tensor<4>) -> MaxPoolOptions<2> {
        let [_batch_size, _channels_in, height_in, width_in] = input.dims();
        let effective_kernel =
            core::array::from_fn(|i| (self.kernel_size[i] - 1) * self.dilation[i] + 1);
        let (padding_height, padding_width) = self.padding.calculate_padding_2d_pairs(
            height_in,
            width_in,
            &effective_kernel,
            &self.stride,
        );

        MaxPoolOptions::new(self.kernel_size)
            .with_stride(self.stride)
            .with_dilation(self.dilation)
            .with_ceil_mode(self.ceil_mode)
            .with_padding_pairs([padding_height, padding_width])
    }

    /// None where the pool cannot size its output, which it rejects.
    fn bounds(&self, input: &Tensor<4>) -> Option<Bounds> {
        let options = self.options(input);
        let output_size = max_pool_output_size(bounds::spatial(input), &options)?;
        Some(
            Windows::max_pool(&options)
                .over(input, output_size)
                .resampling(input),
        )
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// See [max_pool2d](burn::tensor::module::max_pool2d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, height_in, width_in]`
    /// - output: `[batch_size, channels, height_out, width_out]`
    #[compute(bounds = Self::bounds)]
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let options = self.options(&input);
        max_pool2d(input, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// A pool writes the output its windows size, ceil mode included.
    #[cfg(feature = "std")]
    #[test]
    fn a_pool_declares_the_output_it_writes() {
        let device = Default::default();
        let pool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_ceil_mode(true)
            .init();
        let input = Tensor::<4>::zeros([1, 2, 6, 6], &device);
        let read = burn::instrument::tensor_bytes(&input);
        let mut output = None;

        let recorded = crate::recording::recorded(|| {
            output = Some(pool.forward(input));
        });

        assert_eq!(recorded.lines(), ["max pool2d"]);
        assert_eq!(
            recorded.declared(),
            Some(Bounds {
                read,
                write: burn::instrument::tensor_bytes(&output.unwrap()),
                flops: 0,
            })
        );
    }

    #[test]
    fn same_padding_with_dilation() {
        let device = Default::default();
        let input = Tensor::<4>::from_data(
            [[[
                [-9., -1., -8., -2., -7., -3.],
                [-6., -4., -5., -3., -9., -2.],
                [-4., -8., -2., -7., -6., -1.],
                [-8., -2., -7., -1., -3., -4.],
                [-5., -7., -9., -6., -1., -8.],
            ]]],
            &device,
        );
        let pool = MaxPool2dConfig::new([3, 2])
            .with_strides([1, 2])
            .with_dilation([2, 2])
            .with_padding(PaddingConfig2d::Same)
            .init();

        // Effective kernel [5, 3] requires padding (top, bottom) = (2, 2), (left, right) = (0, 1).
        let expected = Tensor::<4>::from_data(
            [[[
                [-2., -2., -6.],
                [-5., -3., -3.],
                [-2., -1., -1.],
                [-5., -3., -3.],
                [-2., -1., -1.],
            ]]],
            &device,
        );
        pool.forward(input)
            .to_data()
            .assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn same_with_even_kernel_uses_asymmetric_padding() {
        let device = Default::default();
        let config = MaxPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Same);
        let pool = config.init();

        // Input: [batch=1, channels=2, height=5, width=5]
        let input = Tensor::<4>::ones([1, 2, 5, 5], &device);
        let output = pool.forward(input);

        // Same padding should preserve spatial dimensions
        assert_eq!(output.dims(), [1, 2, 5, 5]);
    }

    #[test]
    fn display() {
        let config = MaxPool2dConfig::new([3, 3]);

        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "MaxPool2d {kernel_size: [3, 3], stride: [3, 3], padding: Valid, dilation: [1, 1], ceil_mode: false}"
        );
    }

    #[rstest]
    #[case([2, 2])]
    #[case([1, 2])]
    fn default_strides_match_kernel_size(#[case] kernel_size: [usize; 2]) {
        let config = MaxPool2dConfig::new(kernel_size);

        assert_eq!(
            config.strides, kernel_size,
            "Expected strides ({:?}) to match kernel size ({:?}) in default MaxPool2dConfig::new constructor",
            config.strides, config.kernel_size
        );
    }

    #[test]
    fn asymmetric_padding_forward() {
        let device = Default::default();
        // Create max pool with asymmetric padding: top=1, left=2, bottom=3, right=4
        let config = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 2, 3, 4));
        let pool = config.init();

        // Input: [batch=1, channels=2, height=4, width=5]
        let input = Tensor::<4>::ones([1, 2, 4, 5], &device);
        let output = pool.forward(input);

        // Height: 4 + 1 + 3 = 8, output = (8 - 3) / 1 + 1 = 6
        // Width: 5 + 2 + 4 = 11, output = (11 - 3) / 1 + 1 = 9
        assert_eq!(output.dims(), [1, 2, 6, 9]);
    }

    #[test]
    fn symmetric_explicit_padding_forward() {
        let device = Default::default();
        // Create max pool with symmetric explicit padding: top=2, left=2, bottom=2, right=2
        let config = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2));
        let pool = config.init();

        // Input: [batch=1, channels=2, height=4, width=5]
        let input = Tensor::<4>::ones([1, 2, 4, 5], &device);
        let output = pool.forward(input);

        // Height: 4 + 2 + 2 = 8, output = (8 - 3) / 1 + 1 = 6
        // Width: 5 + 2 + 2 = 9, output = (9 - 3) / 1 + 1 = 7
        assert_eq!(output.dims(), [1, 2, 6, 7]);
    }
}
