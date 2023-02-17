use crate as burn;

use crate::config::Config;
use crate::nn::conv::Conv2dPaddingConfig;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::max_pool2d;

/// Configuration to create an [2D max pooling](MaxPool2d) layer.
#[derive(Config)]
pub struct MaxPool2dConfig {
    /// The number of channels.
    pub channels: usize,
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The strides.
    #[config(default = "[1, 1]")]
    pub strides: [usize; 2],
    /// The padding configuration.
    #[config(default = "MaxPool2dPaddingConfig::Valid")]
    pub padding: MaxPool2dPaddingConfig,
}

/// Padding configuration for 2D max pooling [config](MaxPool2dConfig).
pub type MaxPool2dPaddingConfig = Conv2dPaddingConfig;

/// Applies a 2D max pooling over input tensors.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    stride: [usize; 2],
    kernel_size: [usize; 2],
    padding: MaxPool2dPaddingConfig,
}

impl MaxPool2d {
    /// Create the module from the given configuration.
    pub fn new(config: &MaxPool2dConfig) -> Self {
        Self {
            stride: config.strides,
            kernel_size: config.kernel_size,
            padding: config.padding.clone(),
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, channels, height_in, width_in],
    /// - output: [batch_size, channels, height_out, width_out],
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, _channels_in, height_in, width_in] = input.dims();
        let padding =
            self.padding
                .calculate_padding_2d(height_in, width_in, &self.kernel_size, &self.stride);

        max_pool2d(input, self.kernel_size, self.stride, padding)
    }
}
