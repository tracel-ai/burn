use super::QuantizationScheme;

/// Quantized tensor primitive.
pub trait QTensorPrimitive {
    /// Returns the quantization scheme for the given tensor.
    fn scheme(&self) -> &QuantizationScheme;
}
