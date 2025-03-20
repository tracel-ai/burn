use crate::{Int, Tensor, backend::Backend};

/// The tensor quantization parameters.
pub type QuantizationParameters<B> = QParams<Tensor<B, 1>, Tensor<B, 1, Int>>;

/// The quantization tensor data parameters.
#[derive(Clone, Debug)]
pub struct QParams<S, O> {
    /// The scaling factor.
    pub scale: S,
    /// The zero-point offset.
    pub offset: Option<O>,
}

/// The quantization parameters primitive.
///
/// # Remarks
///
/// This is a low-level struct used internally by the library to provide the quantization parameters
/// to the backends. It is not designed for direct usage by users, and not recommended to import
/// or use this struct directly.
///
/// Users should prefer the [QuantizationParameters] struct, which is designed for public use.
pub struct QuantizationParametersPrimitive<B: Backend> {
    /// The scaling factor.
    pub scale: B::FloatTensorPrimitive,
    /// The zero-point offset.
    pub offset: Option<B::IntTensorPrimitive>,
}

impl<B: Backend> From<QuantizationParameters<B>> for QuantizationParametersPrimitive<B> {
    fn from(value: QuantizationParameters<B>) -> Self {
        QuantizationParametersPrimitive {
            scale: value.scale.primitive.tensor(),
            offset: value.offset.map(|x| x.primitive),
        }
    }
}
