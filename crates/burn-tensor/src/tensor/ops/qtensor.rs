use core::future::Future;

use crate::{
    backend::Backend,
    quantization::{QuantizationParametersPrimitive, QuantizationScheme},
    Device, Shape, TensorData,
};

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

    /// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
    fn quantize<const D: usize>(
        tensor: FloatTensor<B, D>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<B>,
    ) -> QuantizedTensor<B, D>;

    /// Convert the tensor back to a higher precision data type.
    fn dequantize<const D: usize>(tensor: QuantizedTensor<B, D>) -> FloatTensor<B, D>;

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
    ) -> impl Future<Output = TensorData> + Send;

    /// Sets the `require_grad` flag of a tensor.
    fn q_set_require_grad<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        _require_grad: bool,
    ) -> QuantizedTensor<B, D> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Returns the `require_grad` flag of a tensor.
    fn q_is_require_grad<const D: usize>(_tensor: &QuantizedTensor<B, D>) -> bool {
        // Should only be overridden by autodiff backends.
        false
    }
}
