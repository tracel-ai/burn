use crate::Backend;

pub use burn_std::quantization::{QParamTensor, QParams};

/// The quantization parameters primitive.
///
/// # Remarks
///
/// This is a low-level struct used internally by the library to provide the quantization parameters
/// to the backends. It is not designed for direct usage by users, and not recommended to import
/// or use this struct directly.
pub struct QuantizationParametersPrimitive<B: Backend> {
    /// The scaling factor.
    pub scales: B::FloatTensorPrimitive,
    /// Optional zero-points for asymmetric quantization.
    /// Used in dequantization: `(q - zero_point) * scale`
    pub zero_points: Option<B::IntTensorPrimitive>,
}
