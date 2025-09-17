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
use burn::tensor::module::conv_transpose3d;
use burn::tensor::ops::ConvTransposeOptions;

/// Configuration to create an [3D transposed convolution](ConvTranspose3d) layer
/// using the [init function](ConvTranspose3dConfig::init).
#[derive(Config, Debug)]
pub struct ConvTranspose3dConfig {
    /// The number of channels.
    pub channels: [usize; 2],
    /// The size of the kernel.
    pub kernel_size: [usize; 3],
    /// The stride of the convolution.
    #[config(default = "[1, 1, 1]")]
    pub stride: [usize; 3],
    /// Spacing between kernel elements.
    #[config(default = "[1, 1, 1]")]
    pub dilation: [usize; 3],
    /// Controls the connections between input and output channels.
    #[config(default = "1")]
    pub groups: usize,
    /// The padding configuration.
    #[config(default = "[0, 0, 0]")]
    pub padding: [usize; 3],
    /// The padding output configuration.
    #[config(default = "[0, 0, 0]")]
    pub padding_out: [usize; 3],
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies a 3D transposed convolution over input tensors.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ConvTranspose3d<B: Backend> {
    /// Tensor of shape `[channels_in, channels_out / groups, kernel_size_1, kernel_size_2, kernel_size_3]`
    pub weight: Param<Tensor<B, 5>>,
    /// Tensor of shape `[channels_out]`
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Stride of the convolution.
    pub stride: [usize; 3],
    /// Size of the kernel.
    pub kernel_size: [usize; 3],
    /// Spacing between kernel elements.
    pub dilation: [usize; 3],
    /// Controls the connections between input and output channels.
    pub groups: usize,
    /// Padding configuration.
    pub padding: [usize; 3],
    /// Padding output configuration.
    pub padding_out: [usize; 3],
    /// Number of channels.
    pub channels: [usize; 2],
}

impl<B: Backend> ModuleDisplay for ConvTranspose3d<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("channels", &format!("{:?}", &self.channels))
            .add("stride", &format!("{:?}", &self.stride))
            .add("kernel_size", &format!("{:?}", &self.kernel_size))
            .add("dilation", &format!("{:?}", &self.dilation))
            .add("groups", &self.groups)
            .add("padding", &format!("{:?}", &self.padding))
            .add("padding_out", &format!("{:?}", &self.padding_out))
            .optional()
    }
}

impl ConvTranspose3dConfig {
    /// Initialize a new [conv transpose 2d](ConvTranspose3d) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTranspose3d<B> {
        checks::checks_channels_div_groups(self.channels[0], self.channels[1], self.groups);

        let shape = [
            self.channels[0],
            self.channels[1] / self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
        ];

        let fan_in = self.channels[1] / self.groups * self.kernel_size.iter().product::<usize>();
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

        ConvTranspose3d {
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

impl<B: Backend> ConvTranspose3d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See also [conv_transpose3d](burn::tensor::module::conv_transpose3d).
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels_in, depth_in, height_in, width_in]`
    /// - output: `[batch_size, channels_out, depth_out, height_out, width_out]`
    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        conv_transpose3d(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvTransposeOptions::new(
                self.stride,
                self.padding,
                self.padding_out,
                self.dilation,
                self.groups,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{ElementConversion, Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;

    #[test]
    fn initializer_default() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = ConvTranspose3dConfig::new([5, 1], [5, 5, 5]);
        let k = (config.channels[1]
            * config.kernel_size[0]
            * config.kernel_size[1]
            * config.kernel_size[2]) as f64;
        let k = (config.groups as f64 / k).sqrt().elem::<FT>();
        let conv = config.init::<TestBackend>(&Default::default());

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config =
            ConvTranspose3dConfig::new([5, 2], [5, 5, 5]).with_initializer(Initializer::Zeros);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight.to_data().assert_approx_eq::<f32>(
            &TensorData::zeros::<f32, _>(conv.weight.shape()),
            Tolerance::default(),
        );
    }

    #[test]
    fn display() {
        let config = ConvTranspose3dConfig::new([5, 2], [5, 5, 5]);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            format!("{conv}"),
            "ConvTranspose3d {channels: [5, 2], stride: [1, 1, 1], kernel_size: [5, 5, 5], dilation: [1, 1, 1], groups: 1, padding: [0, 0, 0], padding_out: [0, 0, 0], params: 1252}"
        );
    }

    #[test]
    #[should_panic = "Number of channels in input tensor and input channels of convolution must be equal. got: 4, expected: 5"]
    fn input_channels_mismatch() {
        let config = ConvTranspose3dConfig::new([5, 3], [3, 3, 3]);
        let conv = config.init::<TestBackend>(&Default::default());

        let input = Tensor::<TestBackend, 5>::zeros([1, 4, 10, 10, 10], &Default::default());
        let _ = conv.forward(input);
    }
}
