use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::ElementConversion;
use crate::tensor::{Distribution, Tensor};
use burn_tensor::module::conv2d;
use burn_tensor::ops::conv::calculate_padding;

/// Configuration to create an [2D convolution](Conv2d) layer.
#[derive(Config)]
pub struct Conv2dConfig {
    /// The number of input channels.
    pub channels_in: usize,
    /// The number of output channels.
    pub channels_out: usize,
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The padding configuration.
    pub padding: Option<Conv2dPaddingConfig>,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
}

/// Padding configuration for 2D convolution [config](Conv2dConfig).
#[derive(Config, Debug)]
pub enum Conv2dPaddingConfig {
    /// Dynamicaly calculate the amount of padding necessary to ensure that the output size will be
    /// the same as the input.
    Same,
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
    bias: Param<Option<Tensor<B, 1>>>,
    stride: [usize; 2],
    kernel_size: [usize; 2],
    padding: Option<Conv2dPaddingConfig>,
}

impl<B: Backend> Conv2d<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &Conv2dConfig) -> Self {
        let k = (config.channels_in * config.kernel_size[0] * config.kernel_size[1]) as f64;
        let k = f64::sqrt(1.0 / k);

        let k1: B::Elem = (-k).to_elem();
        let k2: B::Elem = k.to_elem();

        let weight = Tensor::random(
            [
                config.channels_out,
                config.channels_in,
                config.kernel_size[0],
                config.kernel_size[1],
            ],
            Distribution::Uniform(k1, k2),
        );

        let bias = if config.bias {
            Some(Tensor::random(
                [config.channels_out],
                Distribution::Uniform(k1, k2),
            ))
        } else {
            None
        };

        Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
            stride: [1, 1], // TODO: Add the stride to the configuration when properly supported.
            kernel_size: config.kernel_size,
            padding: config.padding.clone(),
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, channels_in, height_in, width_in],
    /// - output: [batch_size, channels_out, height_out, width_out],
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let same_padding = || {
            let [_batch_size, _channels_in, height_in, width_in] = input.dims();

            let p1 = calculate_padding(self.kernel_size[0], self.stride[0], height_in, height_in);
            let p2 = calculate_padding(self.kernel_size[1], self.stride[1], width_in, width_in);

            [p1, p2]
        };

        let padding = match &self.padding {
            Some(config) => match config {
                Conv2dPaddingConfig::Same => same_padding(),
                Conv2dPaddingConfig::Explicit(v1, v2) => [*v1, *v2],
            },
            None => [0, 0],
        };

        conv2d(
            &input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            padding,
        )
    }
}
