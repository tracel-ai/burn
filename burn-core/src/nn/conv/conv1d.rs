use alloc::format;
use alloc::vec::Vec;

use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::conv1d;
use burn_tensor::ops::conv::calculate_padding;

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
    /// The padding configuration.
    pub padding: Option<Conv1dPaddingConfig>,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::UniformDefault")]
    pub initializer: Initializer,
}

/// Padding configuration for 1D convolution [config](Conv1dConfig).
#[derive(Config, Debug)]
pub enum Conv1dPaddingConfig {
    /// Dynamicaly calculate the amount of padding necessary to ensure that the output size will be
    /// the same as the input.
    Same,
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
    bias: Param<Option<Tensor<B, 1>>>,
    stride: usize,
    kernel_size: usize,
    padding: Option<Conv1dPaddingConfig>,
}

impl<B: Backend> Conv1d<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &Conv1dConfig) -> Self {
        let k = (config.channels_in * config.kernel_size) as f64;
        let k = sqrt(1.0 / k);

        let initializer = if let Initializer::UniformDefault = config.initializer {
            Initializer::Uniform(-k, k)
        } else {
            config.initializer.clone()
        };

        let weight =
            initializer.init([config.channels_out, config.channels_in, config.kernel_size]);

        let bias = if config.bias {
            Some(initializer.init([config.channels_out]))
        } else {
            None
        };

        Self {
            weight: Param::from(weight),
            bias: Param::from(bias),
            stride: 1, // TODO: Add the stride to the configuration when properly supported.
            kernel_size: config.kernel_size,
            padding: config.padding.clone(),
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, channels_in, length_in],
    /// - output: [batch_size, channels_out, length_out],
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let same_padding = || {
            let [_batch_size, _channels_in, length] = input.dims();
            calculate_padding(self.kernel_size, self.stride, length, length)
        };

        let padding = match &self.padding {
            Some(config) => match config {
                Conv1dPaddingConfig::Same => same_padding(),
                Conv1dPaddingConfig::Explicit(value) => *value,
            },
            None => 0,
        };

        conv1d(
            input,
            self.weight.val(),
            self.bias.val(),
            self.stride,
            padding,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub type TB = burn_ndarray::NdArrayBackend<f32>;

    #[test]
    fn initializer_default() {
        TB::seed(0);
        let config = Conv1dConfig::new(5, 5, 5);
        let k = (config.channels_in * config.kernel_size) as f64;
        let k = sqrt(1.0 / k);
        assert_eq!(config.initializer, Initializer::UniformDefault);
        let conv: Conv1d<TB> = Conv1d::new(&config);
        for item in conv.weight.to_data().value.iter() {
            if *item < -k as f32 || *item > k as f32 {
                panic!("Element ({item}) is not within the range of (-{k},{k})");
            }
        }
    }

    #[test]
    fn initializer_zeros() {
        TB::seed(0);
        let config = Conv1dConfig::new(5, 5, 5).with_initializer(Initializer::Zeros);
        assert_eq!(config.initializer, Initializer::Zeros);
        let conv: Conv1d<TB> = Conv1d::new(&config);
        for item in conv.weight.to_data().value.iter() {
            assert_eq!(*item, 0.0f32);
        }
    }
}
