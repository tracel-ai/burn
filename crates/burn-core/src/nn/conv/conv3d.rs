use alloc::format;

use crate as burn;

use crate::config::Config;
use crate::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param};
use crate::nn::Initializer;
use crate::nn::PaddingConfig3d;
use crate::tensor::backend::Backend;
use crate::tensor::module::conv3d;
use crate::tensor::ops::ConvOptions;
use crate::tensor::Tensor;

use crate::nn::conv::checks;

/// Configuration to create a [3D convolution](Conv3d) layer, using the [init function](Conv3dConfig::init).
#[derive(Config, Debug)]
pub struct Conv3dConfig {
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
    #[config(default = "PaddingConfig3d::Valid")]
    pub padding: PaddingConfig3d,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies a 3D convolution over input tensors.
///
/// Should be created with [Conv3dConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Conv3d<B: Backend> {
    /// Tensor of shape `[channels_out, channels_in / groups, kernel_size_1, kernel_size_2, kernel_size_3]`
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
    /// The padding configuration.
    pub padding: Ignored<PaddingConfig3d>,
}

impl Conv3dConfig {
    /// Initialize a new [conv3d](Conv3d) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv3d<B> {
        checks::checks_channels_div_groups(self.channels[0], self.channels[1], self.groups);

        let shape = [
            self.channels[1],
            self.channels[0] / self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
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

        Conv3d {
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

impl<B: Backend> ModuleDisplay for Conv3d<B> {
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

        content
            .add("stride", &stride)
            .add("kernel_size", &kernel_size)
            .add("dilation", &dilation)
            .add("groups", &self.groups)
            .add("padding", &padding_formatted)
            .optional()
    }
}

impl<B: Backend> Conv3d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [conv3d](crate::tensor::module::conv3d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels_in, depth_in, height_in, width_in]`
    /// - output: `[batch_size, channels_out, depth_out, height_out, width_out]`
    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_batch_size, _channels_in, depth_in, height_in, width_in] = input.dims();
        let padding = self.padding.calculate_padding_3d(
            depth_in,
            height_in,
            width_in,
            &self.kernel_size,
            &self.stride,
        );
        conv3d(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvOptions::new(self.stride, padding, self.dilation, self.groups),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorData;
    use crate::TestBackend;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = Conv3dConfig::new([5, 1], [5, 5, 5]);
        let k = (config.channels[0]
            * config.kernel_size[0]
            * config.kernel_size[1]
            * config.kernel_size[2]) as f64;
        let k = (config.groups as f64 / k).sqrt() as f32;
        let device = Default::default();
        let conv = config.init::<TestBackend>(&device);

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = Conv3dConfig::new([5, 2], [5, 5, 5]).with_initializer(Initializer::Zeros);
        let device = Default::default();
        let conv = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight
            .to_data()
            .assert_approx_eq(&TensorData::zeros::<f32, _>(conv.weight.shape()), 3);
    }

    #[test]
    fn initializer_fan_out() {
        TestBackend::seed(0);

        let init = Initializer::KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: true, // test that fan_out is passed to `init_with()`
        };
        let device = Default::default();
        let config = Conv3dConfig::new([5, 1], [5, 5, 5]).with_initializer(init.clone());
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
        let config = Conv3dConfig::new([4, 4], [1, 1, 1])
            .with_initializer(init.clone())
            .with_groups(4);
        let _ = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, init);
    }

    #[test]
    fn display() {
        let config = Conv3dConfig::new([5, 1], [5, 5, 5]);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{}", conv),
            "Conv3d {stride: [1, 1, 1], kernel_size: [5, 5, 5], dilation: [1, 1, 1], groups: 1, padding: Valid, params: 626}"
        );
    }
}
