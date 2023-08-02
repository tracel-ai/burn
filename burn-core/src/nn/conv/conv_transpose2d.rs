use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::conv_transpose2d;
use burn_tensor::ops::ConvTransposeOptions;

/// Configuration to create an [2D transposed convolution](ConvTranspose2d) layer.
#[derive(Config, Debug)]
pub struct ConvTranspose2dConfig {
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
    #[config(default = "[0, 0]")]
    pub padding: [usize; 2],
    /// The padding output configuration.
    #[config(default = "[0, 0]")]
    pub padding_out: [usize; 2],
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters.
    /// Setting this parameter will override the default initialization scheme.
    ///
    /// # Default initialization
    ///
    /// - weight: Tensor initialized from a uniform distribution
    ///           `U(-k, k)` where `k = sqrt(groups / (channels_out * kernel_size_1 * kernel_size_2))`
    ///
    /// - bias:   Tensor initialized from a uniform distribution
    ///           `U(-k, k)` where `k = sqrt(groups / (channels_out * kernel_size_1 * kernel_size_2))`
    pub initializer: Option<Initializer>,
}

/// Applies a 2D transposed convolution over input tensors.
///
/// # Params
///
/// - weight: Tensor of shape `[channels_in, channels_out / groups, kernel_size_1, kernel_size_2]`
///
/// - bias:   Tensor of shape `[channels_out]`
#[derive(Module, Debug)]
pub struct ConvTranspose2d<B: Backend> {
    weight: Param<Tensor<B, 4>>,
    bias: Option<Param<Tensor<B, 1>>>,
    stride: [usize; 2],
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    padding: [usize; 2],
    padding_out: [usize; 2],
}

impl ConvTranspose2dConfig {
    /// Initialize a new [conv transpose 2d](ConvTranspose2d) module.
    pub fn init<B: Backend>(&self) -> ConvTranspose2d<B> {
        let shape = [
            self.channels[0],
            self.channels[1] / self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        ];
        let fan_in = self.channels[1] * self.kernel_size.iter().product::<usize>();

        let weight = match &self.initializer {
            Some(initializer) => initializer.init_with(shape, Some(fan_in), None),
            None => {
                let k = libm::sqrt(self.groups as f64 / fan_in as f64);
                Initializer::Uniform { min: -k, max: k }.init(shape)
            }
        };

        let mut bias = None;

        if self.bias {
            bias = Some(match &self.initializer {
                Some(initializer) => initializer.init_with([self.channels[1]], Some(fan_in), None),
                None => {
                    let k = libm::sqrt(self.groups as f64 / fan_in as f64);
                    Initializer::Uniform { min: -k, max: k }.init([self.channels[1]])
                }
            });
        }

        ConvTranspose2d {
            weight: Param::from(weight),
            bias: bias.map(Param::from),
            stride: self.stride,
            kernel_size: self.kernel_size,
            dilation: self.dilation,
            groups: self.groups,
            padding: self.padding,
            padding_out: self.padding_out,
        }
    }

    /// Initialize a new [conv2d](Conv2d) module with a [record](Conv2dRecord).
    pub fn init_with<B: Backend>(&self, record: ConvTranspose2dRecord<B>) -> ConvTranspose2d<B> {
        ConvTranspose2d {
            weight: record.weight,
            bias: record.bias,
            stride: self.stride,
            dilation: self.dilation,
            kernel_size: self.kernel_size,
            groups: self.groups,
            padding: self.padding,
            padding_out: self.padding_out,
        }
    }
}

impl<B: Backend> ConvTranspose2d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, channels_in, height_in, width_in],
    /// - output: [batch_size, channels_out, height_out, width_out],
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        conv_transpose2d(
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
    use burn_tensor::Data;

    use super::*;
    use crate::TestBackend;
    use libm::sqrt;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = ConvTranspose2dConfig::new([5, 1], [5, 5]).with_groups(2);
        let k = (config.channels[1] * config.kernel_size[0] * config.kernel_size[1]) as f64;
        let k = sqrt(config.groups as f64 / k) as f32;
        let conv = config.init::<TestBackend>();

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config =
            ConvTranspose2dConfig::new([5, 2], [5, 5]).with_initializer(Some(Initializer::Zeros));
        let conv = config.init::<TestBackend>();

        assert_eq!(config.initializer, Some(Initializer::Zeros));
        conv.weight
            .to_data()
            .assert_approx_eq(&Data::zeros(conv.weight.shape()), 3);
    }
}
