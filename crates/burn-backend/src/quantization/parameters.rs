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
    /// The scaling factor, one per block or a single one for a per-tensor level, in the quantized
    /// tensor's float dtype.
    pub scales: B::FloatTensorPrimitive,
    /// The per-tensor scale that [`scales`](Self::scales) are expressed relative to, for a
    /// two-level scheme. Always `f32`, unlike `scales`, so the two cannot go into a binary op
    /// without a cast.
    pub global: Option<B::FloatTensorPrimitive>,
}
