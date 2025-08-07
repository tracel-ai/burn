use super::QuantSettings;

/// Quantized tensor primitive.
pub trait QTensorPrimitive {
    /// Returns the quantization settings for the given tensor.
    fn settings(&self) -> &QuantSettings;
}
