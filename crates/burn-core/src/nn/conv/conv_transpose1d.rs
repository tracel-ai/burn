use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::conv::checks;
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::module::conv_transpose1d;
use crate::tensor::ops::ConvTransposeOptions;
use crate::tensor::Tensor;

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
pub struct ConvTranspose1d<B: Backend> {
    /// Tensor of shape `[channels_in, channels_out / groups, kernel_size]`
    pub weight: Param<Tensor<B, 3>>,
    /// Tensor of shape `[channels_out]`
    pub bias: Option<Param<Tensor<B, 1>>>,
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    padding: usize,
    padding_out: usize,
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
        }
    }
}

impl<B: Backend> ConvTranspose1d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See also [conv_transpose1d](crate::tensor::module::conv_transpose1d).
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
    use super::*;
    use crate::tensor::Data;
    use crate::TestBackend;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = ConvTranspose1dConfig::new([5, 1], 5);
        let k = (config.channels[1] * config.kernel_size) as f64;
        let k = (config.groups as f64 / k).sqrt() as f32;
        let conv = config.init::<TestBackend>(&Default::default());

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = ConvTranspose1dConfig::new([5, 2], 5).with_initializer(Initializer::Zeros);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight
            .to_data()
            .assert_approx_eq(&Data::zeros(conv.weight.shape()), 3);
    }
}
