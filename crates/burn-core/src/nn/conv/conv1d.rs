use alloc::format;

use crate as burn;

use crate::{
    config::Config,
    module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param},
    nn::{Initializer, PaddingConfig1d, conv::checks},
    tensor::{Tensor, backend::Backend, module::conv1d, ops::ConvOptions},
};

/// Configuration to create a [1D convolution](Conv1d) layer using the [init function](Conv1dConfig::init).
#[derive(Config, Debug)]
pub struct Conv1dConfig {
    /// The number of input channels.
    pub channels_in: usize,
    /// The number of output channels.
    pub channels_out: usize,
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The stride of the convolution.
    #[config(default = "1")]
    pub stride: usize,
    /// Spacing between kernel elements.
    #[config(default = "1")]
    pub dilation: usize,
    /// Controls the connections between input and output channels.
    #[config(default = "1")]
    pub groups: usize,
    /// The padding configuration.
    ///
    /// ### Warning
    /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
    /// size is not supported as it will not produce the same output size.
    #[config(default = "PaddingConfig1d::Valid")]
    pub padding: PaddingConfig1d,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies a 1D convolution over input tensors.
///
/// Should be created with [Conv1dConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Conv1d<B: Backend> {
    /// Tensor of shape `[channels_out, channels_in / groups, kernel_size]`
    pub weight: Param<Tensor<B, 3>>,
    /// Tensor of shape `[channels_out]`
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Stride of the convolution.
    pub stride: usize,
    /// Size of the kernel.
    pub kernel_size: usize,
    /// Spacing between kernel elements.
    pub dilation: usize,
    /// Controls the connections between input and output channels.
    pub groups: usize,
    /// Padding configuration.
    pub padding: Ignored<PaddingConfig1d>,
}

impl<B: Backend> ModuleDisplay for Conv1d<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        // Since padding does not implement ModuleDisplay, we need to format it manually.
        let padding_formatted = format!("{}", &self.padding);

        content
            .add("stride", &self.stride)
            .add("kernel_size", &self.kernel_size)
            .add("dilation", &self.dilation)
            .add("groups", &self.groups)
            .add("padding", &padding_formatted)
            .optional()
    }
}

impl Conv1dConfig {
    /// Initialize a new [conv1d](Conv1d) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv1d<B> {
        checks::checks_channels_div_groups(self.channels_in, self.channels_out, self.groups);
        if self.padding == PaddingConfig1d::Same {
            checks::check_same_padding_support(&[self.kernel_size]);
        }

        let shape = [
            self.channels_out,
            self.channels_in / self.groups,
            self.kernel_size,
        ];

        let fan_in: usize = self.channels_in / self.groups * self.kernel_size;
        let weight = self
            .initializer
            .init_with(shape, Some(fan_in), None, device);
        let mut bias = None;

        if self.bias {
            bias =
                Some(
                    self.initializer
                        .init_with([self.channels_out], Some(fan_in), None, device),
                );
        }

        Conv1d {
            weight,
            bias,
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}

impl<B: Backend> Conv1d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [conv1d](crate::tensor::module::conv1d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels_in, length_in]`
    /// - output: `[batch_size, channels_out, length_out]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let length = input.dims()[2];
        let padding = self
            .padding
            .calculate_padding_1d(length, self.kernel_size, self.stride);

        conv1d(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvOptions::new([self.stride], [padding], [self.dilation], self.groups),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::tensor::TensorData;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = Conv1dConfig::new(5, 5, 5);
        let k = (config.channels_in * config.kernel_size) as f64;
        let k = (config.groups as f64 / k).sqrt() as f32;
        let conv = config.init::<TestBackend>(&Default::default());

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = Conv1dConfig::new(5, 5, 5).with_initializer(Initializer::Zeros);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight
            .to_data()
            .assert_approx_eq(&TensorData::zeros::<f32, _>(conv.weight.shape()), 3);
    }

    #[test]
    #[should_panic = "Same padding with an even kernel size is not supported"]
    fn same_with_even_kernel_is_invalid() {
        let device = Default::default();
        let config = Conv1dConfig::new(5, 5, 4).with_padding(PaddingConfig1d::Same);
        let _ = config.init::<TestBackend>(&device);
    }

    #[test]
    fn display() {
        let config = Conv1dConfig::new(5, 5, 5);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{}", conv),
            "Conv1d {stride: 1, kernel_size: 5, dilation: 1, groups: 1, padding: Valid, params: 130}"
        );
    }

    #[test]
    #[should_panic = "Number of channels in input tensor and input channels of convolution must be equal. got: 4, expected: 5"]
    fn input_channels_mismatch() {
        let config = Conv1dConfig::new(5, 3, 3);
        let conv = config.init::<TestBackend>(&Default::default());

        let input = Tensor::<TestBackend, 3>::zeros([1, 4, 10], &Default::default());
        let _ = conv.forward(input);
    }
}
