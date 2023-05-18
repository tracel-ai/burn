use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::conv::Conv1dPaddingConfig;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::avg_pool1d;

/// Configuration to create a [1D avg pooling](AvgPool1d) layer.
#[derive(Config)]
pub struct AvgPool1dConfig {
    /// The number of channels.
    pub channels: usize,
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The stride.
    #[config(default = "1")]
    pub stride: usize,
    /// The padding configuration.
    #[config(default = "AvgPool1dPaddingConfig::Valid")]
    pub padding: AvgPool1dPaddingConfig,
}

/// Padding configuration for 1D avg pooling [config](AvgPool1dConfig).
pub type AvgPool1dPaddingConfig = Conv1dPaddingConfig;

/// Applies a 1D avg pooling over input tensors.
#[derive(Module, Debug, Clone)]
pub struct AvgPool1d {
    stride: usize,
    kernel_size: usize,
    padding: AvgPool1dPaddingConfig,
}

impl AvgPool1dConfig {
    /// Initialize a new [avg pool 1d](AvgPool1d) module.
    pub fn init(&self) -> AvgPool1d {
        AvgPool1d {
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
        }
    }
}

impl AvgPool1d {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, channels, length_in],
    /// - output: [batch_size, channels, length_out],
    pub fn forward<B: Backend>(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch_size, _channels, length] = input.dims();
        let padding = self
            .padding
            .calculate_padding_1d(length, self.kernel_size, self.stride);

        avg_pool1d(input, self.kernel_size, self.stride, padding)
    }
}
