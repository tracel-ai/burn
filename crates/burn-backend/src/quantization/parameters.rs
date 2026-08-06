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
    /// two-level scheme. A value is reconstructed as `q * global * scale`.
    ///
    /// Always `f32`, whatever the tensor's float dtype is, so it does not match
    /// [`scales`](Self::scales) and the two cannot go into a binary op without a cast. It sits a
    /// whole block param's range below the block scales by construction, which at `f16` would put
    /// it among the subnormals. That guarantee is what lets `validate_levels` require an `f32`
    /// stored param.
    pub global: Option<B::FloatTensorPrimitive>,
}
