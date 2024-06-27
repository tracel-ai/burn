use crate::{backend::QuantizationBackend, QuantizationStrategy, Tensor, TensorData};

impl<const D: usize, B> Tensor<B, D>
where
    B: QuantizationBackend,
{
    /// Convert the tensor to a lower precision data type based on the quantization strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The quantization strategy.
    ///
    /// # Returns
    ///
    /// The quantized tensor.
    pub fn quantize(self, strategy: QuantizationStrategy) -> QTensor<B, D> {
        QTensor::Quantized {
            tensor: B::quantize(self.primitive, &strategy),
            strategy,
        }
    }
}

/// A quantized tensor with a given quantization backend and shape.
#[derive(Debug, Clone)]
pub enum QTensor<B: QuantizationBackend, const D: usize> {
    /// Quantized tensor representation.
    Quantized {
        /// The underlying quantized tensor.
        tensor: B::QuantizedTensorPrimitive<D>,
        /// The tensor quantization strategy.
        strategy: QuantizationStrategy,
    },
    /// Not quantized tensor representation.
    NotQuantized(B::FloatTensorPrimitive<D>),
}

impl<B: QuantizationBackend, const D: usize> QTensor<B, D> {
    /// Returns the full tensor representation.
    pub fn tensor(self) -> Tensor<B, D> {
        match self {
            Self::Quantized { tensor, strategy } => Tensor::new(B::dequantize(tensor, &strategy)),
            Self::NotQuantized(tensor) => Tensor::new(tensor),
        }
    }

    /// Returns the data of the current tensor.
    pub fn into_data(self) -> TensorData {
        match self {
            Self::Quantized {
                tensor,
                strategy: _,
            } => B::quantized_into_data(tensor).read(),
            Self::NotQuantized(tensor) => B::float_into_data(tensor).read(),
        }
    }
}
