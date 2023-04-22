use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::conv2d;
use burn_tensor::ops::conv::calculate_conv_padding;
use burn_tensor::ops::ConvOptions;

use libm::sqrt;

/// Configuration to create an [2D convolution](Conv2d) layer.
#[derive(Config)]
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
    #[config(default = "Conv2dPaddingConfig::Valid")]
    pub padding: Conv2dPaddingConfig,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::UniformDefault")]
    pub initializer: Initializer,
}

/// Padding configuration for 2D convolution [config](Conv2dConfig).
#[derive(Module, Config, Debug)]
pub enum Conv2dPaddingConfig {
    /// Dynamicaly calculate the amount of padding necessary to ensure that the output size will be
    /// the same as the input.
    Same,
    /// Same as no padding.
    Valid,
    /// Applies the specified amount of padding to all inputs.
    Explicit(usize, usize),
}

/// Applies a 2D convolution over input tensors.
///
/// # Params
///
/// - weight: Tensor of shape [channels_out, channels_in, kernel_size_1, kernel_size_2] initialized from a uniform
///     distribution `U(-k, k)` where `k = sqrt(1 / channels_in * kernel_size_1 * kernel_size_2)`
///
/// - bias:   Tensor of shape [channels_out], initialized from a uniform distribution `U(-k, k)`
///     where `k = sqrt(1 / channels_in * kernel_size_1 * kernel_size_2)`
#[derive(Module, Debug)]
pub struct Conv2d<B: Backend> {
    weight: Param<Tensor<B, 4>>,
    bias: Option<Param<Tensor<B, 1>>>,
    stride: [usize; 2],
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    padding: Conv2dPaddingConfig,
}

impl Conv2dConfig {
    /// Initialize a new [conv2d](Conv2d) module.
    pub fn init<B: Backend>(&self) -> Conv2d<B> {
        let k = (self.channels[0] * self.kernel_size[0] * self.kernel_size[1]) as f64;
        let k = sqrt(1.0 / k);

        let initializer = if let Initializer::UniformDefault = self.initializer {
            Initializer::Uniform(-k, k)
        } else {
            self.initializer.clone()
        };

        let weight = initializer.init([
            self.channels[1],
            self.channels[0],
            self.kernel_size[0],
            self.kernel_size[1],
        ]);

        let bias = if self.bias {
            Some(initializer.init([self.channels[1]]))
        } else {
            None
        };

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

impl Conv2dPaddingConfig {
    pub(crate) fn calculate_padding_2d(
        &self,
        height: usize,
        width: usize,
        kernel_size: &[usize; 2],
        stride: &[usize; 2],
    ) -> [usize; 2] {
        let same_padding = || {
            let p1 = calculate_conv_padding(kernel_size[0], stride[0], height, height);
            let p2 = calculate_conv_padding(kernel_size[1], stride[1], width, width);

            [p1, p2]
        };

        match self {
            Conv2dPaddingConfig::Same => same_padding(),
            Conv2dPaddingConfig::Valid => [0, 0],
            Conv2dPaddingConfig::Explicit(v1, v2) => [*v1, *v2],
        }
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::Data;

    use super::*;
    use crate::TestBackend;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = Conv2dConfig::new([5, 1], [5, 5]);
        let k = (config.channels[0] * config.kernel_size[0] * config.kernel_size[1]) as f64;
        let k = sqrt(1.0 / k) as f32;
        let conv = config.init::<TestBackend>();

        assert_eq!(config.initializer, Initializer::UniformDefault);
        conv.weight.to_data().assert_in_range(-k, k);
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
