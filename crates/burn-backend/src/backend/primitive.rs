use crate::Backend;
use burn_std::quantization::{QuantAcc, QuantPropagation, QuantScheme};
use burn_std::{DType, Shape};

#[derive(Debug, Clone)]
/// A primitive tensor representation.
pub enum TensorPrimitive<B: Backend> {
    /// Float tensor primitive.
    Float(B::FloatTensorPrimitive),
    /// Quantized float tensor primitive.
    QFloat(B::QuantizedTensorPrimitive),
}

impl<B: Backend> TensorPrimitive<B> {
    /// Returns the full tensor representation.
    pub fn tensor(self) -> B::FloatTensorPrimitive {
        match self {
            Self::QFloat(tensor) => B::dequantize(tensor),
            Self::Float(tensor) => tensor,
        }
    }

    /// Returns a mutable reference to the full tensor representation.
    pub fn get_mut_ref(&mut self) -> &mut B::FloatTensorPrimitive {
        match self {
            // Self::QFloat(tensor) => B::dequantize(tensor),
            Self::QFloat(_tensor) => todo!(),
            Self::Float(tensor) => tensor,
        }
    }
}

impl<B: Backend> TensorMetadata for TensorPrimitive<B> {
    fn dtype(&self) -> DType {
        match self {
            TensorPrimitive::Float(tensor) => tensor.dtype(),
            TensorPrimitive::QFloat(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            TensorPrimitive::Float(tensor) => tensor.shape(),
            TensorPrimitive::QFloat(tensor) => tensor.shape(),
        }
    }

    fn rank(&self) -> usize {
        match self {
            TensorPrimitive::Float(tensor) => tensor.rank(),
            TensorPrimitive::QFloat(tensor) => tensor.rank(),
        }
    }
}

/// Tensor metadata trait for tensor primitive.
pub trait TensorMetadata: Clone + Send + Sync + core::fmt::Debug {
    /// The dtype of the tensor.
    fn dtype(&self) -> DType;
    /// The shape of the tensor.
    fn shape(&self) -> Shape;

    /// The number of dimensions of the tensor.
    fn rank(&self) -> usize {
        self.shape().num_dims()
    }
}

/// Quantized tensor primitive.
pub trait QTensorPrimitive {
    /// Returns the quantization settings for the given tensor.
    fn scheme(&self) -> &QuantScheme;
    /// The precision used for the accumulation in various kernels.
    fn acc_precision(&self) -> QuantAcc {
        QuantAcc::F32
    }
    /// How quantization is propagated during computation.
    fn propagation(&self) -> QuantPropagation {
        QuantPropagation::Inhibit
    }

    /// Returns the default tensor quantization scheme.
    fn default_scheme() -> QuantScheme {
        QuantScheme::default()
    }
}
