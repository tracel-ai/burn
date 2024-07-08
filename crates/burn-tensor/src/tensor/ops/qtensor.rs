use crate::{backend::Backend, Device, QuantizationStrategy, Shape};

use super::{FloatTensor, QuantizedTensor};

/// Quantized Tensor API for basic operations, see [tensor](crate::Tensor)
/// for documentation on each function.
pub trait QTensorOps<B: Backend> {
    /// Convert the tensor to a lower precision data type based on the quantization strategy.
    fn quantize<const D: usize>(
        tensor: FloatTensor<B, D>,
        strategy: &QuantizationStrategy,
    ) -> QuantizedTensor<B, D>;

    /// Convert the tensor back to a higher precision data type based on the quantization strategy.
    fn dequantize<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        strategy: &QuantizationStrategy,
    ) -> FloatTensor<B, D>;

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn q_shape<const D: usize>(tensor: &QuantizedTensor<B, D>) -> Shape<D>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn q_device<const D: usize>(tensor: &QuantizedTensor<B, D>) -> Device<B>;
}
