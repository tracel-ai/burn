use core::future::Future;

use crate::{backend::Backend, Device, QuantizationStrategy, Shape, TensorData};

use super::{FloatTensor, QuantizedTensor};

/// Quantized Tensor API for basic operations, see [tensor](crate::Tensor)
/// for documentation on each function.
pub trait QTensorOps<B: Backend> {
    /// Creates a new tensor from the data structure.
    ///
    /// # Arguments
    ///
    /// * `data` - The data structure.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given data.
    fn q_from_data<const D: usize>(data: TensorData, device: &Device<B>) -> QuantizedTensor<B, D>;

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

    /// Reshapes a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reshape.
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the new shape.
    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<B, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<B, D2>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn q_into_data<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        strategy: QuantizationStrategy,
    ) -> impl Future<Output = TensorData> + Send;
}
