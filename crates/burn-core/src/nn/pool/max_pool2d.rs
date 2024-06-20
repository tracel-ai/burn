use crate as burn;

use crate::config::Config;
use crate::module::{Ignored, Module};
use crate::nn::PaddingConfig2d;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use crate::tensor::module::max_pool2d;

/// Configuration to create a [2D max pooling](MaxPool2d) layer using the [init function](MaxPool2dConfig::init).
#[derive(Debug, Config)]
pub struct MaxPool2dConfig {
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The strides.
    #[config(default = "[1, 1]")]
    pub strides: [usize; 2],
    /// The padding configuration.
    #[config(default = "PaddingConfig2d::Valid")]
    pub padding: PaddingConfig2d,
    /// The dilation.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
}

/// Applies a 2D max pooling over input tensors.
///
/// Should be created with [MaxPool2dConfig](MaxPool2dConfig).
#[derive(Module, Clone, Debug)]
pub struct MaxPool2d {
    stride: [usize; 2],
    kernel_size: [usize; 2],
    padding: Ignored<PaddingConfig2d>,
    dilation: [usize; 2],
}

impl MaxPool2dConfig {
    /// Initialize a new [max pool 2d](MaxPool2d) module.
    pub fn init(&self) -> MaxPool2d {
        MaxPool2d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: Ignored(self.padding.clone()),
            dilation: self.dilation,
        }
    }
}

impl MaxPool2d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [max_pool2d](crate::tensor::module::max_pool2d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, height_in, width_in]`
    /// - output: `[batch_size, channels, height_out, width_out]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, _channels_in, height_in, width_in] = input.dims();
        let padding =
            self.padding
                .calculate_padding_2d(height_in, width_in, &self.kernel_size, &self.stride);

        max_pool2d(input, self.kernel_size, self.stride, padding, self.dilation)
    }
}
