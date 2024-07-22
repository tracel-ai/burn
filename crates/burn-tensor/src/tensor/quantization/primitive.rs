use super::{QuantizationScheme, QuantizationStrategy};

/// Quantized tensor primitive.
pub trait QTensorPrimitive {
    /// Returns the quantization scheme for the given tensor.
    fn scheme(&self) -> &QuantizationScheme;
    /// Returns the quantization strategy for the given tensor.
    ///
    /// # Remarks
    /// Retrieving the quantization strategy with its corresponding parameters might require
    /// synchronization on the backend.
    fn strategy(&self) -> QuantizationStrategy;
}
