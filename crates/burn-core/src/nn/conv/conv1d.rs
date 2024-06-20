use alloc::format;

use crate as burn;

use crate::{
    config::Config,
    module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param},
    nn::{conv::checks, Initializer, PaddingConfig1d},
    tensor::{backend::Backend, module::conv1d, ops::ConvOptions, Tensor},
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
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    padding: Ignored<PaddingConfig1d>,
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
        let [_batch_size, _channels, length] = input.dims();
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
    use crate::tensor::Data;
    use crate::TestBackend;

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
            .assert_approx_eq(&Data::zeros(conv.weight.shape()), 3);
    }
}
