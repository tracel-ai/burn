use crate::{DType, Shape, Tensor, backend::Backend};
use alloc::vec::Vec;

/// The tensor quantization parameters.
pub type QuantizationParameters<B> = QParams<Tensor<B, 1>>;

/// The quantization tensor data parameters.
#[derive(Clone, Debug)]
pub struct QParams<S> {
    /// The scaling factor.
    pub scales: S,
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
    pub scales: B::FloatTensorPrimitive,
}

impl<B: Backend> From<QuantizationParameters<B>> for QuantizationParametersPrimitive<B> {
    fn from(value: QuantizationParameters<B>) -> Self {
        QuantizationParametersPrimitive {
            scales: value.scales.primitive.tensor(),
        }
    }
}

/// A quantization parameter tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QParamTensor {
    /// Start of the tensor in the buffer
    pub offset_start: usize,
    /// Offset of tensor end from the end of the buffer
    pub offset_end: usize,
    /// Shape of the tensor
    pub shape: Shape,
    /// Strides of the tensor
    pub strides: Vec<usize>,
    /// Data type of the tensor
    pub dtype: DType,
}
