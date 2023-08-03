use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::Initializer;
use crate::nn::PaddingConfig2d;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::conv2d;
use burn_tensor::ops::ConvOptions;
use libm::sqrt;

use super::checks;

/// Configuration to create an [2D convolution](Conv2d) layer.
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
    #[config(default = "PaddingConfig2d::Valid")]
    pub padding: PaddingConfig2d,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::KaimingUniform{gain:1.0/sqrt(3.0),fan_out_only:false}")]
    pub initializer: Initializer,
}

/// Applies a 2D convolution over input tensors.
///
/// # Params
///
/// - weight: Tensor of shape `[channels_out, channels_in / groups, kernel_size_1, kernel_size_2]`
///
/// - bias:   Tensor of shape `[channels_out]`
#[derive(Module, Debug)]
pub struct Conv2d<B: Backend> {
    weight: Param<Tensor<B, 4>>,
    bias: Option<Param<Tensor<B, 1>>>,
    stride: [usize; 2],
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    padding: PaddingConfig2d,
}

impl Conv2dConfig {
    /// Initialize a new [conv2d](Conv2d) module.
    pub fn init<B: Backend>(&self) -> Conv2d<B> {
        checks::checks_channels_div_groups(self.channels[0], self.channels[1], self.groups);

        let shape = [
            self.channels[1],
            self.channels[0] / self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        ];

        let fan_in = self.channels[0] / self.groups * self.kernel_size.iter().product::<usize>();
        let weight = self.initializer.init_with(shape, Some(fan_in), None);
        let mut bias = None;

        if self.bias {
            bias = Some(
                self.initializer
                    .init_with([self.channels[1]], Some(fan_in), None),
            );
        }

        Conv2d {
            weight: Param::from(weight),
            bias: bias.map(Param::from),
            stride: self.stride,
            kernel_size: self.kernel_size,
            dilation: self.dilation,
            padding: self.padding.clone(),
            groups: self.groups,
        }
    }

    /// Initialize a new [conv2d](Conv2d) module with a [record](Conv2dRecord).
    pub fn init_with<B: Backend>(&self, record: Conv2dRecord<B>) -> Conv2d<B> {
        Conv2d {
            weight: record.weight,
            bias: record.bias,
            stride: self.stride,
            dilation: self.dilation,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
            groups: self.groups,
        }
    }
}

impl<B: Backend> Conv2d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, channels_in, height_in, width_in],
    /// - output: [batch_size, channels_out, height_out, width_out],
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
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Data;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = Conv2dConfig::new([5, 1], [5, 5]);
        let k = (config.channels[0] * config.kernel_size[0] * config.kernel_size[1]) as f64;
        let k = sqrt(config.groups as f64 / k) as f32;
        let conv = config.init::<TestBackend>();

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = Conv2dConfig::new([5, 2], [5, 5]).with_initializer(Initializer::Zeros);
        let conv = config.init::<TestBackend>();

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight
            .to_data()
            .assert_approx_eq(&Data::zeros(conv.weight.shape()), 3);
    }
}
