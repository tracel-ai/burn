use super::QuantScheme;

/// Quantized tensor primitive.
pub trait QTensorPrimitive {
    /// Returns the quantization scheme for the given tensor.
    fn scheme(&self) -> &QuantScheme;
}
