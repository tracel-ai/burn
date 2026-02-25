use alloc::format;

use burn_core as burn;

use crate::{PaddingConfig1d, conv::checks};
use burn::tensor::{Tensor, backend::Backend, module::conv1d, ops::PaddedConvOptions};
use burn::{
    config::Config,
    module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay, Param},
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
    /// Supports symmetric and asymmetric padding. `Same` padding with even kernel sizes
    /// will automatically use asymmetric padding to preserve input dimensions.
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
    #[module(skip)]
    pub padding: PaddingConfig1d,
}

impl<B: Backend> ModuleDisplay for Conv1d<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        // Format stride/dilation as strings
        let stride = format!("{:?}", self.stride);
        let kernel_size = format!("{:?}", self.kernel_size);
        let dilation = format!("{:?}", self.dilation);

        // Extract channels in/out from weight dims
        let [channels_out, group_channels_in, _] = self.weight.dims();
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
            .add_debug_attribute("padding", &self.padding)
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
            padding: self.padding.clone(),
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}

impl<B: Backend> Conv1d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [conv1d](burn::tensor::module::conv1d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels_in, length_in]`
    /// - output: `[batch_size, channels_out, length_out]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let length = input.dims()[2];

        // Calculate padding as pair - handles Same, Valid, and Explicit uniformly
        let (left, right) =
            self.padding
                .calculate_padding_1d_pair(length, self.kernel_size, self.stride);

        let options = PaddedConvOptions::asymmetric(
            [self.stride],
            [left],
            [right],
            [self.dilation],
            self.groups,
        );

        conv1d(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            options,
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{ElementConversion, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;

    #[test]
    fn initializer_default() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = Conv1dConfig::new(5, 5, 5);
        let k = (config.channels_in * config.kernel_size) as f64;
        let k = (config.groups as f64 / k).sqrt().elem::<FT>();
        let conv = config.init::<TestBackend>(&device);

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = Conv1dConfig::new(5, 5, 5).with_initializer(Initializer::Zeros);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight
            .to_data()
            .assert_eq(&TensorData::zeros::<FT, _>(conv.weight.shape()), false);
    }

    #[test]
    fn same_with_even_kernel_uses_asymmetric_padding() {
        let device = Default::default();
        let config = Conv1dConfig::new(4, 4, 2)
            .with_padding(PaddingConfig1d::Same)
            .with_initializer(Initializer::Constant { value: 1.0 })
            .with_bias(false);
        let conv = config.init::<TestBackend>(&device);

        // Input: [batch=1, channels=4, length=5]
        let input = Tensor::<TestBackend, 3>::ones([1, 4, 5], &device);
        let output = conv.forward(input);

        // Same padding should preserve spatial dimensions
        assert_eq!(output.dims(), [1, 4, 5]);
    }

    #[test]
    fn display() {
        let config = Conv1dConfig::new(5, 5, 5);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{conv}"),
            "Conv1d {ch_in: 5, ch_out: 5, stride: 1, kernel_size: 5, dilation: 1, groups: 1, padding: Valid, params: 130}"
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

    #[test]
    fn asymmetric_padding_forward() {
        let device = Default::default();
        // Create conv with asymmetric padding: left=1, right=2
        let config = Conv1dConfig::new(2, 3, 3)
            .with_padding(PaddingConfig1d::Explicit(1, 2))
            .with_initializer(Initializer::Constant { value: 1.0 })
            .with_bias(false);
        let conv = config.init::<TestBackend>(&device);

        // Input: [batch=1, channels=2, length=4]
        let input = Tensor::<TestBackend, 3>::ones([1, 2, 4], &device);
        let output = conv.forward(input);

        // With asymmetric padding (1, 2), input length 4 becomes 4+1+2=7
        // Output length = (7 - 3) / 1 + 1 = 5
        assert_eq!(output.dims(), [1, 3, 5]);
    }

    #[test]
    fn symmetric_explicit_padding_forward() {
        let device = Default::default();
        // Create conv with symmetric explicit padding: left=2, right=2
        let config = Conv1dConfig::new(2, 3, 3)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_initializer(Initializer::Constant { value: 1.0 })
            .with_bias(false);
        let conv = config.init::<TestBackend>(&device);

        // Input: [batch=1, channels=2, length=4]
        let input = Tensor::<TestBackend, 3>::ones([1, 2, 4], &device);
        let output = conv.forward(input);

        // With symmetric padding (2, 2), input length 4 becomes 4+2+2=8
        // Output length = (8 - 3) / 1 + 1 = 6
        assert_eq!(output.dims(), [1, 3, 6]);
    }
}
