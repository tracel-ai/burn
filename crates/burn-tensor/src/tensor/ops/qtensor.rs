use burn_common::reader::Reader;

use crate::{backend::QuantizationBackend, QuantizationStrategy, TensorData};

/// Quantized Tensor operations API, see [tensor](crate::Tensor) and [quantized tensor][crate::QTensor]
/// for documentation on each function.
pub trait QTensorOps<B: QuantizationBackend> {
    /// Convert the tensor to a lower precision data type based on the quantization strategy.
    fn quantize<const D: usize>(
        tensor: B::FloatTensorPrimitive<D>,
        strategy: &QuantizationStrategy,
    ) -> B::QuantizedTensorPrimitive<D>;

    /// Convert the tensor back to a higher precision data type based on the quantization strategy.
    fn dequantize<const D: usize>(
        tensor: B::QuantizedTensorPrimitive<D>,
        strategy: &QuantizationStrategy,
    ) -> B::FloatTensorPrimitive<D>;

    /// Returns the data of the quantized tensor.
    fn quantized_into_data<const D: usize>(
        tensor: B::QuantizedTensorPrimitive<D>,
    ) -> Reader<TensorData>;
}
