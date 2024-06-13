use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use crate::tensor::module::adaptive_avg_pool1d;

/// Configuration to create a [1D adaptive avg pooling](AdaptiveAvgPool1d) layer using the [init function](AdaptiveAvgPool1dConfig::init).
#[derive(Config)]
pub struct AdaptiveAvgPool1dConfig {
    /// The size of the output.
    pub output_size: usize,
}

/// Applies a 1D adaptive avg pooling over input tensors.
///
/// Should be created with [AdaptiveAvgPool1dConfig].
#[derive(Module, Clone, Debug)]
pub struct AdaptiveAvgPool1d {
    output_size: usize,
}

impl AdaptiveAvgPool1dConfig {
    /// Initialize a new [adaptive avg pool 1d](AdaptiveAvgPool1d) module.
    pub fn init(&self) -> AdaptiveAvgPool1d {
        AdaptiveAvgPool1d {
            output_size: self.output_size,
        }
    }
}

impl AdaptiveAvgPool1d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [adaptive_avg_pool1d](crate::tensor::module::adaptive_avg_pool1d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, length]`
    /// - output: `[batch_size, channels, length_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        adaptive_avg_pool1d(input, self.output_size)
    }
}
