use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use crate::tensor::module::adaptive_avg_pool2d;

/// Configuration to create a [2D adaptive avg pooling](AdaptiveAvgPool2d) layer using the [init function](AdaptiveAvgPool2dConfig::init).
#[derive(Config)]
pub struct AdaptiveAvgPool2dConfig {
    /// The size of the output.
    pub output_size: [usize; 2],
}

/// Applies a 2D adaptive avg pooling over input tensors.
///
/// Should be created with [AdaptiveAvgPool2dConfig].
#[derive(Module, Clone, Debug)]
pub struct AdaptiveAvgPool2d {
    output_size: [usize; 2],
}

impl AdaptiveAvgPool2dConfig {
    /// Initialize a new [adaptive avg pool 2d](AdaptiveAvgPool2d) module.
    pub fn init(&self) -> AdaptiveAvgPool2d {
        AdaptiveAvgPool2d {
            output_size: self.output_size,
        }
    }
}

impl AdaptiveAvgPool2d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [adaptive_avg_pool2d](crate::tensor::module::adaptive_avg_pool2d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, height_in, width_in]`
    /// - output: `[batch_size, channels, height_out, width_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        adaptive_avg_pool2d(input, self.output_size)
    }
}
