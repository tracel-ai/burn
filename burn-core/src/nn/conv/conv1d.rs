use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::conv1d;
use burn_tensor::ops::conv::calculate_conv_padding;
use burn_tensor::ops::ConvOptions;

use libm::sqrt;

/// Configuration to create an [1D convolution](Conv1d) layer.
#[derive(Config)]
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
    #[config(default = "Conv1dPaddingConfig::Valid")]
    pub padding: Conv1dPaddingConfig,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::UniformDefault")]
    pub initializer: Initializer,
}

/// Padding configuration for 1D convolution [config](Conv1dConfig).
#[derive(Module, Config, Debug)]
pub enum Conv1dPaddingConfig {
    /// Dynamicaly calculate the amount of padding necessary to ensure that the output size will be
    /// the same as the input.
    Same,
    /// Same as no padding.
    Valid,
    /// Applies the specified amount of padding to all inputs.
    Explicit(usize),
}

/// Applies a 1D convolution over input tensors.
///
/// # Params
///
/// - weight: Tensor of shape [channels_out, channels_in, kernel_size] initialized from a uniform
///     distribution `U(-k, k)` where `k = sqrt(1 / channels_in * kernel_size)`
///
/// - bias:   Tensor of shape [channels_out], initialized from a uniform distribution `U(-k, k)`
///     where `k = sqrt(1 / channels_in * kernel_size)`
#[derive(Module, Debug)]
pub struct Conv1d<B: Backend> {
    weight: Param<Tensor<B, 3>>,
    bias: Option<Param<Tensor<B, 1>>>,
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    padding: Conv1dPaddingConfig,
}

impl Conv1dConfig {
    /// Initialize a new [conv1d](Conv1d) module.
    pub fn init<B: Backend>(&self) -> Conv1d<B> {
        let k = (self.channels_in * self.kernel_size) as f64;
        let k = sqrt(1.0 / k);

        let initializer = if let Initializer::UniformDefault = self.initializer {
            Initializer::Uniform(-k, k)
        } else {
            self.initializer.clone()
        };

        let weight = initializer.init([self.channels_out, self.channels_in, self.kernel_size]);

        let bias = if self.bias {
            Some(Param::from(initializer.init([self.channels_out])))
        } else {
            None
        };

        Conv1d {
            weight: Param::from(weight),
            bias,
            stride: 1, // TODO: Add the stride to the config when properly supported.
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
            dilation: self.dilation,
            groups: self.groups,
        }
    }
    /// Initialize a new [conv1d](Conv1d) module with a [record](Conv1dRecord).
    pub fn init_with<B: Backend>(&self, record: Conv1dRecord<B>) -> Conv1d<B> {
        Conv1d {
            weight: record.weight,
            bias: record.bias,
            stride: 1, // TODO: Add the stride to the config when properly supported.
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
    /// # Shapes
    ///
    /// - input: [batch_size, channels_in, length_in],
    /// - output: [batch_size, channels_out, length_out],
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

impl Conv1dPaddingConfig {
    pub(crate) fn calculate_padding_1d(
        &self,
        length: usize,
        kernel_size: usize,
        stride: usize,
    ) -> usize {
        let same_padding = || calculate_conv_padding(kernel_size, stride, length, length);

        match self {
            Conv1dPaddingConfig::Valid => 0,
            Conv1dPaddingConfig::Same => same_padding(),
            Conv1dPaddingConfig::Explicit(value) => *value,
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

        let config = Conv1dConfig::new(5, 5, 5);
        let k = (config.channels_in * config.kernel_size) as f64;
        let k = sqrt(1.0 / k) as f32;
        let conv = config.init::<TestBackend>();

        assert_eq!(config.initializer, Initializer::UniformDefault);
        conv.weight.to_data().assert_in_range(-k, k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = Conv1dConfig::new(5, 5, 5).with_initializer(Initializer::Zeros);
        let conv = config.init::<TestBackend>();

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight
            .to_data()
            .assert_approx_eq(&Data::zeros(conv.weight.shape()), 3);
    }
}
