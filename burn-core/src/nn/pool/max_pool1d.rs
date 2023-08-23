use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::PaddingConfig1d;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::max_pool1d;

/// Configuration to create a [1D max pooling](MaxPool1d) layer.
#[derive(Config)]
pub struct MaxPool1dConfig {
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The stride.
    #[config(default = "1")]
    pub stride: usize,
    /// The padding configuration.
    #[config(default = "PaddingConfig1d::Valid")]
    pub padding: PaddingConfig1d,
    /// The dilation.
    #[config(default = "1")]
    pub dilation: usize,
}

/// Applies a 1D max pooling over input tensors.
#[derive(Module, Debug, Clone)]
pub struct MaxPool1d {
    stride: usize,
    kernel_size: usize,
    padding: PaddingConfig1d,
    dilation: usize,
}

impl MaxPool1dConfig {
    /// Initialize a new [max pool 1d](MaxPool1d) module.
    pub fn init(&self) -> MaxPool1d {
        MaxPool1d {
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
            dilation: self.dilation,
        }
    }
}

impl MaxPool1d {
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

        max_pool1d(input, self.kernel_size, self.stride, padding, self.dilation)
    }
}
