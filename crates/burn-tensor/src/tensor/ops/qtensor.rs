use crate::{backend::Backend, QuantizationStrategy};

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
}
