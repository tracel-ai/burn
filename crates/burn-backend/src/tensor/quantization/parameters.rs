use crate::Backend;
use alloc::vec::Vec;
use burn_std::{DType, Shape};

pub use burn_std::quantization::QParams;

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
