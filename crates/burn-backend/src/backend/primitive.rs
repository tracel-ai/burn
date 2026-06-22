use crate::{Backend, BackendTypes, DeviceOps, get_device_settings};
use burn_std::{DType, QuantScheme, Shape};

#[derive(Debug, Clone)]
/// A primitive tensor representation.
pub enum TensorPrimitive<B: BackendTypes> {
    /// Float tensor primitive.
    Float(B::FloatTensorPrimitive),
    /// Quantized float tensor primitive.
    QFloat(B::QuantizedTensorPrimitive),
}

impl<B: Backend> TensorPrimitive<B> {
    /// Returns the full tensor representation.
    pub fn tensor(self) -> B::FloatTensorPrimitive {
        match self {
            Self::QFloat(tensor) => {
                let dtype = get_device_settings::<B>(&tensor.device()).float_dtype;
                B::dequantize(tensor, dtype)
            }
            Self::Float(tensor) => tensor,
        }
    }

    /// Returns a mutable reference to the full tensor representation.
    pub fn get_mut_ref(&mut self) -> &mut B::FloatTensorPrimitive {
        match self {
            Self::QFloat(_tensor) => todo!(),
            Self::Float(tensor) => tensor,
        }
    }
}

impl<B: BackendTypes> TensorMetadata for TensorPrimitive<B> {
    type Device = B::Device;

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
    fn device(&self) -> Self::Device {
        match self {
            TensorPrimitive::Float(tensor) => tensor.device(),
            TensorPrimitive::QFloat(tensor) => tensor.device(),
        }
    }
}

/// Tensor metadata trait for tensor primitive.
pub trait TensorMetadata: Clone + Send + Sync + core::fmt::Debug {
    /// The device type associated with the tensor.
    type Device: DeviceOps;
    /// Get the dtype of the tensor.
    fn dtype(&self) -> DType;
    /// Get the shape of the tensor.
    fn shape(&self) -> Shape;

    /// Get the number of dimensions of the tensor.
    fn rank(&self) -> usize {
        self.shape().num_dims()
    }
    /// Get the device associated with the tensor.
    fn device(&self) -> Self::Device;

    /// Get the [quantization scheme](QuantScheme) for a quantized float tensor.
    ///
    /// # Panics
    /// Panics if the tensor is not quantized.
    fn scheme(&self) -> QuantScheme {
        match self.dtype() {
            DType::QFloat(scheme) => scheme,
            other => panic!("Quantization scheme is not valid for dtype {other:?}"),
        }
    }
}
