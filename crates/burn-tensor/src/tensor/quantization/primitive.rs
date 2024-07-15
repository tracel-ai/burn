use super::QuantizationStrategy;

/// Quantized tensor primitive.
pub trait QTensorPrimitive {
    /// Returns the quantization strategy for the given tensor.
    fn strategy(&self) -> QuantizationStrategy;
}
