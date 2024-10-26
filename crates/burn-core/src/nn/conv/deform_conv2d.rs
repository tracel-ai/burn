use alloc::format;
use burn_tensor::ops::DeformConvOptions;

use crate as burn;

use crate::config::Config;
use crate::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param};
use crate::nn::Initializer;
use crate::nn::PaddingConfig2d;
use crate::tensor::backend::Backend;
use crate::tensor::module::deform_conv2d;
use crate::tensor::Tensor;

use crate::nn::conv::checks;

/// Configuration to create a [deformable 2D convolution](DeformConv2d) layer, using the [init function](DeformConv2dConfig::init).
#[derive(Config, Debug)]
pub struct DeformConv2dConfig {
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
    pub weight_groups: usize,
    /// Offset groups.
    #[config(default = "1")]
    pub offset_groups: usize,
    /// The padding configuration.
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

/// Applies a deformable 2D convolution over input tensors.
///
/// Should be created with [DeformConv2dConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct DeformConv2d<B: Backend> {
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
    pub weight_groups: usize,
    /// Offset groups.
    pub offset_groups: usize,
    /// The padding configuration.
    pub padding: Ignored<PaddingConfig2d>,
}

impl DeformConv2dConfig {
    /// Initialize a new [DeformConv2d](DeformConv2d) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeformConv2d<B> {
        checks::checks_channels_div_groups(self.channels[0], self.channels[1], self.weight_groups);

        let shape = [
            self.channels[1],
            self.channels[0] / self.weight_groups,
            self.kernel_size[0],
            self.kernel_size[1],
        ];

        let k = self.kernel_size.iter().product::<usize>();
        let fan_in = self.channels[0] / self.weight_groups * k;
        let fan_out = self.channels[1] / self.weight_groups * k;

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

        DeformConv2d {
            weight,
            bias,
            stride: self.stride,
            kernel_size: self.kernel_size,
            dilation: self.dilation,
            padding: Ignored(self.padding.clone()),
            weight_groups: self.weight_groups,
            offset_groups: self.weight_groups,
        }
    }
}

impl<B: Backend> ModuleDisplay for DeformConv2d<B> {
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
            .add("weight_groups", &self.weight_groups)
            .add("offset_groups", &self.offset_groups)
            .add("padding", &padding_formatted)
            .optional()
    }
}

impl<B: Backend> DeformConv2d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [deform_conv2d](crate::tensor::module::deform_conv2d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels_in, height_in, width_in]`
    /// - offset: `[batch_size, 2 * offset_groups * kernel_height * kernel_width, height_out, width_out]`
    /// - mask: `[batch_size, offset_groups * kernel_height * kernel_width, height_out, width_out]`
    /// - output: `[batch_size, channels_out, height_out, width_out]`
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        offset: Tensor<B, 4>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let [_batch_size, _channels_in, height_in, width_in] = input.dims();
        let padding =
            self.padding
                .calculate_padding_2d(height_in, width_in, &self.kernel_size, &self.stride);
        deform_conv2d(
            input,
            offset,
            self.weight.val(),
            mask,
            self.bias.as_ref().map(|bias| bias.val()),
            DeformConvOptions::new(
                self.stride,
                padding,
                self.dilation,
                self.weight_groups,
                self.offset_groups,
            ),
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

        let config = DeformConv2dConfig::new([5, 1], [5, 5]);
        let k = (config.channels[0] * config.kernel_size[0] * config.kernel_size[1]) as f64;
        let k = (config.offset_groups as f64 / k).sqrt() as f32;
        let device = Default::default();
        let conv = config.init::<TestBackend>(&device);

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = DeformConv2dConfig::new([5, 2], [5, 5]).with_initializer(Initializer::Zeros);
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
        let config = DeformConv2dConfig::new([5, 1], [5, 5]).with_initializer(init.clone());
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
        let config = DeformConv2dConfig::new([4, 4], [1, 1])
            .with_initializer(init.clone())
            .with_weight_groups(4);
        let _ = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, init);
    }

    #[test]
    #[should_panic = "Both channels must be divisible by the number of groups."]
    fn channels_with_groups_is_invalid() {
        let device = Default::default();
        let config = DeformConv2dConfig::new([1, 4], [1, 1]).with_weight_groups(4);
        let _ = config.init::<TestBackend>(&device);
    }

    #[test]
    fn display() {
        let config = DeformConv2dConfig::new([5, 1], [5, 5]);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{}", conv),
            "DeformConv2d {stride: [1, 1], kernel_size: [5, 5], dilation: [1, 1], weight_groups: 1, offset_groups: 1, padding: Valid, params: 126}"
        );
    }
}
