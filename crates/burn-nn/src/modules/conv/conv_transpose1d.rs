use alloc::format;

use burn_core as burn;

use crate::conv::checks;
use burn::config::Config;
use burn::module::Content;
use burn::module::DisplaySettings;
use burn::module::Initializer;
use burn::module::Module;
use burn::module::ModuleDisplay;
use burn::module::Param;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::module::conv_transpose1d;
use burn::tensor::ops::ConvTransposeOptions;

/// Configuration to create an [1D transposed convolution](ConvTranspose1d) layer
/// using the [init function](ConvTranspose1dConfig::init).
#[derive(Config, Debug)]
pub struct ConvTranspose1dConfig {
    /// The number of channels.
    pub channels: [usize; 2],
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
    #[config(default = "0")]
    pub padding: usize,
    /// The padding output configuration.
    #[config(default = "0")]
    pub padding_out: usize,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies a 1D transposed convolution over input tensors.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ConvTranspose1d<B: Backend> {
    /// Tensor of shape `[channels_in, channels_out / groups, kernel_size]`
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
    /// The padding configuration.
    pub padding: usize,
    /// The padding output configuration.
    pub padding_out: usize,
    /// The number of channels.
    pub channels: [usize; 2],
}

impl<B: Backend> ModuleDisplay for ConvTranspose1d<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("channels", &format!("{:?}", &self.channels))
            .add("stride", &self.stride)
            .add("kernel_size", &self.kernel_size)
            .add("dilation", &self.dilation)
            .add("groups", &self.groups)
            .add("padding", &self.padding)
            .add("padding_out", &self.padding_out)
            .optional()
    }
}

impl ConvTranspose1dConfig {
    /// Initialize a new [conv transpose 1d](ConvTranspose1d) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTranspose1d<B> {
        checks::checks_channels_div_groups(self.channels[0], self.channels[1], self.groups);

        let shape = [
            self.channels[0],
            self.channels[1] / self.groups,
            self.kernel_size,
        ];

        let fan_in = self.channels[1] / self.groups * self.kernel_size;
        let weight = self
            .initializer
            .init_with(shape, Some(fan_in), None, device);
        let mut bias = None;

        if self.bias {
            bias = Some(
                self.initializer
                    .init_with([self.channels[1]], Some(fan_in), None, device),
            );
        }

        ConvTranspose1d {
            weight,
            bias,
            stride: self.stride,
            kernel_size: self.kernel_size,
            dilation: self.dilation,
            groups: self.groups,
            padding: self.padding,
            padding_out: self.padding_out,
            channels: self.channels,
        }
    }
}

impl<B: Backend> ConvTranspose1d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See also [conv_transpose1d](burn::tensor::module::conv_transpose1d).
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels_in, length_in]`
    /// - output: `[batch_size, channels_out, length_out]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        conv_transpose1d(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvTransposeOptions::new(
                [self.stride],
                [self.padding],
                [self.padding_out],
                [self.dilation],
                self.groups,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::ops::FloatElem;
    use burn::tensor::{ElementConversion, Tolerance};

    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn initializer_default() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = ConvTranspose1dConfig::new([5, 1], 5);
        let k = (config.channels[1] * config.kernel_size) as f64;
        let k = (config.groups as f64 / k).sqrt().elem::<FT>();
        let conv = config.init::<TestBackend>(&Default::default());

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = ConvTranspose1dConfig::new([5, 2], 5).with_initializer(Initializer::Zeros);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight.to_data().assert_approx_eq::<f32>(
            &TensorData::zeros::<f32, _>(conv.weight.shape()),
            Tolerance::default(),
        );
    }

    #[test]
    fn display() {
        let config = ConvTranspose1dConfig::new([5, 2], 5);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            format!("{conv}"),
            "ConvTranspose1d {channels: [5, 2], stride: 1, kernel_size: 5, dilation: 1, groups: 1, padding: 0, padding_out: 0, params: 52}"
        );
    }

    #[test]
    #[should_panic = "Number of channels in input tensor and input channels of convolution must be equal. got: 4, expected: 5"]
    fn input_channels_mismatch() {
        let config = ConvTranspose1dConfig::new([5, 3], 3);
        let conv = config.init::<TestBackend>(&Default::default());

        let input = Tensor::<TestBackend, 3>::zeros([1, 4, 10], &Default::default());
        let _ = conv.forward(input);
    }
}
