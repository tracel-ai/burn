use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::PaddingConfig1d;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::module::avg_pool1d;

/// Configuration to create a [1D avg pooling](AvgPool1d) layer.
#[derive(Config)]
pub struct AvgPool1dConfig {
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The stride.
    #[config(default = "1")]
    pub stride: usize,
    /// The padding configuration.
    #[config(default = "PaddingConfig1d::Valid")]
    pub padding: PaddingConfig1d,
    /// If the padding is counted in the denominator when computing the average.
    #[config(default = "true")]
    count_include_pad: bool,
}

/// Applies a 1D avg pooling over input tensors.
///
/// See [AvgPool1dConfig](AvgPool1dConfig) for details.
///
/// # Remarks
///
/// The zero-padding values will be included in the calculation
/// of the average. This means that the zeros are counted as
/// legitimate values, and they contribute to the denominator
/// when calculating the average. This is equivalent to
/// `torch.nn.AvgPool2d` with `count_include_pad=True`.
///
/// TODO: Add support for `count_include_pad=False`, see
/// [Issue 636](https://github.com/burn-rs/burn/issues/636)

#[derive(Module, Debug, Clone)]
pub struct AvgPool1d {
    stride: usize,
    kernel_size: usize,
    padding: PaddingConfig1d,
    count_include_pad: bool,
}

impl AvgPool1dConfig {
    /// Initialize a new [avg pool 1d](AvgPool1d) module.
    pub fn init(&self) -> AvgPool1d {
        AvgPool1d {
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
            count_include_pad: self.count_include_pad,
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

        avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            padding,
            self.count_include_pad,
        )
    }
}
