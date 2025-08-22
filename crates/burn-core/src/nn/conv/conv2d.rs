use alloc::format;

use crate as burn;

use crate::config::Config;
use crate::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param};
use crate::nn::Initializer;
use crate::nn::PaddingConfig2d;
use crate::tensor::Tensor;
use crate::tensor::backend::Backend;
use crate::tensor::module::conv2d;
use crate::tensor::ops::ConvOptions;

use crate::nn::conv::checks;

/// Configuration to create a [2D convolution](Conv2d) layer, using the [init function](Conv2dConfig::init).
#[derive(Config, Debug)]
pub struct Conv2dConfig {
    /// The number of channels.
    pub channels: [usize; 2],
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The stride of the convolution.
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    /// Spacing between kernel elements.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
    /// Controls the connections between input and output channels.
    #[config(default = "1")]
    pub groups: usize,
    /// The padding configuration.
    ///
    /// ### Warning
    /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
    /// size is not supported as it will not produce the same output size.
    #[config(default = "PaddingConfig2d::Valid")]
    pub padding: PaddingConfig2d,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies a 2D convolution over input tensors.
///
/// Should be created with [Conv2dConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Conv2d<B: Backend> {
    /// Tensor of shape `[channels_out, channels_in / groups, kernel_size_1, kernel_size_2]`
    pub weight: Param<Tensor<B, 4>>,
    /// Tensor of shape `[channels_out]`
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Stride of the convolution.
    pub stride: [usize; 2],
    /// Size of the kernel.
    pub kernel_size: [usize; 2],
    /// Spacing between kernel elements.
    pub dilation: [usize; 2],
    /// Controls the connections between input and output channels.
    pub groups: usize,
    /// The padding configuration.
    pub padding: Ignored<PaddingConfig2d>,
}

impl Conv2dConfig {
    /// Initialize a new [conv2d](Conv2d) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2d<B> {
        checks::checks_channels_div_groups(self.channels[0], self.channels[1], self.groups);
        if self.padding == PaddingConfig2d::Same {
            checks::check_same_padding_support(&self.kernel_size);
        }

        let shape = [
            self.channels[1],
            self.channels[0] / self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        ];

        let k = self.kernel_size.iter().product::<usize>();
        let fan_in = self.channels[0] / self.groups * k;
        let fan_out = self.channels[1] / self.groups * k;

        let weight = self
            .initializer
            .init_with(shape, Some(fan_in), Some(fan_out), device);
        let mut bias = None;

        if self.bias {
            bias = Some(self.initializer.init_with(
                [self.channels[1]],
                Some(fan_in),
                Some(fan_out),
                device,
            ));
        }

        Conv2d {
            weight,
            bias,
            stride: self.stride,
            kernel_size: self.kernel_size,
            dilation: self.dilation,
            padding: Ignored(self.padding.clone()),
            groups: self.groups,
        }
    }
}

impl<B: Backend> ModuleDisplay for Conv2d<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        // Since padding does not implement ModuleDisplay, we need to format it manually.
        let padding_formatted = format!("{}", &self.padding);

        // Format the stride, kernel_size and dilation as strings, formatted as arrays instead of indexed.
        let stride = format!("{:?}", self.stride);
        let kernel_size = format!("{:?}", self.kernel_size);
        let dilation = format!("{:?}", self.dilation);
        let [channels_out, group_channels_in, _, _] = self.weight.dims();
        let channels_in = group_channels_in * self.groups;
        let ch_out = format!("{:?}", channels_out);
        let ch_in = format!("{:?}", channels_in);
        content
            .add("ch_in", &ch_in)
            .add("ch_out", &ch_out)
            .add("stride", &stride)
            .add("kernel_size", &kernel_size)
            .add("dilation", &dilation)
            .add("groups", &self.groups)
            .add("padding", &padding_formatted)
            .optional()
    }
}

impl<B: Backend> Conv2d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [conv2d](crate::tensor::module::conv2d) for more information.
    ///
    /// # Shapes
    /// - `input`: `[batch_size, channels_in, height_in, width_in]`
    /// - `output`: `[batch_size, channels_out, height_out, width_out]`
    ///
    /// # Example
    /// ```rust,ignore
    /// use burn::nn::conv::Conv2dConfig;
    /// use burn::tensor::Tensor;
    ///
    /// // Assuming backend type alias `B`
    /// let device = Default::default();
    /// let conv = Conv2dConfig::new([3, 8], [3, 3]).init::<B>(&device);
    ///
    /// let x = Tensor::<B, 4>::zeros([1, 3, 28, 28], &device);
    /// let y = conv.forward(x);
    ///
    /// println!("{:?}", y.dims()); // [1, 8, 26, 26]
    /// ```
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, _channels_in, height_in, width_in] = input.dims();
        let padding =
            self.padding
                .calculate_padding_2d(height_in, width_in, &self.kernel_size, &self.stride);
        conv2d(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvOptions::new(self.stride, padding, self.dilation, self.groups),
        )
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::ops::FloatElem;
    use burn_tensor::{ElementConversion, Tolerance};

    use super::*;
    use crate::TestBackend;
    use crate::tensor::TensorData;
    type FT = FloatElem<TestBackend>; // Float test

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = Conv2dConfig::new([5, 1], [5, 5]);
        let k = (config.channels[0] * config.kernel_size[0] * config.kernel_size[1]) as f64;
        let k = (config.groups as f64 / k).sqrt().elem::<FT>();
        let device = Default::default();
        let conv = config.init::<TestBackend>(&device);

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = Conv2dConfig::new([5, 2], [5, 5]).with_initializer(Initializer::Zeros);
        let device = Default::default();
        let conv = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight.to_data().assert_approx_eq::<FT>(
            &TensorData::zeros::<FT, _>(conv.weight.shape()),
            Tolerance::default(),
        );
    }

    #[test]
    fn initializer_fan_out() {
        TestBackend::seed(0);

        let init = Initializer::KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: true, // test that fan_out is passed to `init_with()`
        };
        let device = Default::default();
        let config = Conv2dConfig::new([5, 1], [5, 5]).with_initializer(init.clone());
        let _ = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, init);
    }

    #[test]
    fn initializer_fan_with_groups_is_valid() {
        TestBackend::seed(0);

        let init = Initializer::KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: true,
        };
        let device = Default::default();
        let config = Conv2dConfig::new([4, 4], [1, 1])
            .with_initializer(init.clone())
            .with_groups(4);
        let _ = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, init);
    }

    #[test]
    #[should_panic = "Both channels must be divisible by the number of groups."]
    fn channels_with_groups_is_invalid() {
        let device = Default::default();
        let config = Conv2dConfig::new([1, 4], [1, 1]).with_groups(4);
        let _ = config.init::<TestBackend>(&device);
    }

    #[test]
    #[should_panic = "Same padding with an even kernel size is not supported"]
    fn same_with_even_kernel_is_invalid() {
        let device = Default::default();
        let config = Conv2dConfig::new([4, 4], [2, 2]).with_padding(PaddingConfig2d::Same);
        let _ = config.init::<TestBackend>(&device);
    }

    #[test]
    fn display() {
        let config = Conv2dConfig::new([5, 1], [5, 5]);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{conv}"),
            "Conv2d {ch_in: 5, ch_out: 1, stride: [1, 1], kernel_size: [5, 5], dilation: [1, 1], groups: 1, padding: Valid, params: 126}"
        );
    }

    #[test]
    #[should_panic = "Number of channels in input tensor and input channels of convolution must be equal. got: 4, expected: 5"]
    fn input_channels_mismatch() {
        let config = Conv2dConfig::new([5, 3], [3, 3]);
        let conv = config.init::<TestBackend>(&Default::default());

        let input = Tensor::<TestBackend, 4>::zeros([1, 4, 10, 10], &Default::default());
        let _ = conv.forward(input);
    }
}
