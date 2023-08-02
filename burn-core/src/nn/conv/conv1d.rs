use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::{Initializer, PaddingConfig1d};
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::conv1d;
use burn_tensor::ops::ConvOptions;

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
    #[config(default = "PaddingConfig1d::Valid")]
    pub padding: PaddingConfig1d,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters.
    /// Setting this parameter will override the default initialization scheme.
    ///
    /// # Default initialization
    ///
    /// - weight: Tensor initialized from a uniform distribution
    ///           `U(-k, k)` where `k = sqrt(groups / (channels_in * kernel_size))`
    ///
    /// - bias:   Tensor initialized from a uniform distribution
    ///           `U(-k, k)` where `k = sqrt(groups / (channels_in * kernel_size))`
    pub initializer: Option<Initializer>,
}

/// Applies a 1D convolution over input tensors.
///
/// # Params
///
/// - weight: Tensor of shape [channels_out, channels_in / groups, kernel_size]
///
/// - bias:   Tensor of shape `[channels_out]`
#[derive(Module, Debug)]
pub struct Conv1d<B: Backend> {
    weight: Param<Tensor<B, 3>>,
    bias: Option<Param<Tensor<B, 1>>>,
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    padding: PaddingConfig1d,
}

impl Conv1dConfig {
    /// Initialize a new [conv1d](Conv1d) module.
    pub fn init<B: Backend>(&self) -> Conv1d<B> {
        let shape = [
            self.channels_out,
            self.channels_in / self.groups,
            self.kernel_size,
        ];
        let fan_in: usize = self.channels_in * self.kernel_size;

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
                Some(initializer) => initializer.init_with([self.channels_out], Some(fan_in), None),
                None => {
                    let k = libm::sqrt(self.groups as f64 / fan_in as f64);
                    Initializer::Uniform { min: -k, max: k }.init([self.channels_out])
                }
            });
        }

        Conv1d {
            weight: Param::from(weight),
            bias: bias.map(Param::from),
            stride: self.stride,
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

#[cfg(test)]
mod tests {
    use burn_tensor::Data;
    use libm::sqrt;

    use super::*;
    use crate::TestBackend;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = Conv1dConfig::new(5, 5, 5);
        let k = (config.channels_in * config.kernel_size) as f64;
        let k = sqrt(config.groups as f64 / k) as f32;
        let conv = config.init::<TestBackend>();

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = Conv1dConfig::new(5, 5, 5).with_initializer(Some(Initializer::Zeros));
        let conv = config.init::<TestBackend>();

        assert_eq!(config.initializer, Some(Initializer::Zeros));
        conv.weight
            .to_data()
            .assert_approx_eq(&Data::zeros(conv.weight.shape()), 3);
    }
}
