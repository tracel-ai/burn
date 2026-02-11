use burn_backend::{Backend, QTensorPrimitive, TensorMetadata};

use crate::backends::*;

// TODO: if we reduce the different associated types for float/int/bool/quantized tensor primitives down to a single
// `B::TensorPrimitive` we can simplify this.

/// Tensor which points to a backend tensor primitive kind.
#[derive(Clone, Debug)]
pub enum BackendTensor<B: Backend> {
    /// Float tensor handle.
    Float(B::FloatTensorPrimitive),
    /// Int tensor handle.
    Int(B::IntTensorPrimitive),
    /// Bool tensor handle.
    Bool(B::BoolTensorPrimitive),
    /// Quantized tensor handle.
    Quantized(B::QuantizedTensorPrimitive),
}

impl<B: Backend> BackendTensor<B> {
    /// Returns the inner float tensor primitive.
    pub(crate) fn float(self) -> B::FloatTensorPrimitive {
        match self {
            BackendTensor::Float(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    /// Returns the inner int tensor primitive.
    pub(crate) fn int(self) -> B::IntTensorPrimitive {
        match self {
            BackendTensor::Int(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    /// Returns the inner bool tensor primitive.
    pub(crate) fn bool(self) -> B::BoolTensorPrimitive {
        match self {
            BackendTensor::Bool(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    /// Returns the inner quantized tensor primitive.
    pub(crate) fn quantized(self) -> B::QuantizedTensorPrimitive {
        match self {
            BackendTensor::Quantized(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    /// Returns the backend device.
    pub(crate) fn device(&self) -> B::Device {
        match self {
            BackendTensor::Float(tensor) => B::float_device(tensor),
            BackendTensor::Int(tensor) => B::int_device(tensor),
            BackendTensor::Bool(tensor) => B::bool_device(tensor),
            BackendTensor::Quantized(tensor) => B::q_device(tensor),
        }
    }
}

pub(crate) fn as_float<B: Backend>(tensor: B::FloatTensorPrimitive) -> BackendTensor<B> {
    BackendTensor::Float(tensor)
}

pub(crate) fn as_qfloat<B: Backend>(tensor: B::QuantizedTensorPrimitive) -> BackendTensor<B> {
    BackendTensor::Quantized(tensor)
}

pub(crate) fn as_int<B: Backend>(tensor: B::IntTensorPrimitive) -> BackendTensor<B> {
    BackendTensor::Int(tensor)
}

pub(crate) fn as_bool<B: Backend>(tensor: B::BoolTensorPrimitive) -> BackendTensor<B> {
    BackendTensor::Bool(tensor)
}

impl<B: Backend> TensorMetadata for BackendTensor<B> {
    fn dtype(&self) -> burn_std::DType {
        match self {
            BackendTensor::Float(tensor) => tensor.dtype(),
            BackendTensor::Int(tensor) => tensor.dtype(),
            BackendTensor::Bool(tensor) => tensor.dtype(),
            BackendTensor::Quantized(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> burn_std::Shape {
        match self {
            BackendTensor::Float(tensor) => tensor.shape(),
            BackendTensor::Int(tensor) => tensor.shape(),
            BackendTensor::Bool(tensor) => tensor.shape(),
            BackendTensor::Quantized(tensor) => tensor.shape(),
        }
    }
}

impl<B: Backend> QTensorPrimitive for BackendTensor<B> {
    fn scheme(&self) -> &burn_std::QuantScheme {
        match self {
            BackendTensor::Quantized(tensor) => tensor.scheme(),
            _ => panic!(
                "Quantization scheme is not valid for dtype {:?}",
                self.dtype(),
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub enum EngineTensor {
    /// The [CPU backend](Cpu) tensor.
    #[cfg(feature = "cpu")]
    Cpu(BackendTensor<Cpu>),

    /// The [CUDA backend](Cuda) tensor.
    #[cfg(feature = "cuda")]
    Cuda(BackendTensor<Cuda>),

    /// The [Metal backend](Metal) tensor.
    #[cfg(feature = "metal")]
    Metal(BackendTensor<Metal>),

    /// The [ROCm backend](Rocm) tensor.
    #[cfg(feature = "rocm")]
    Rocm(BackendTensor<Rocm>),

    /// The [Vulkan backend](Vulkan) tensor.
    #[cfg(feature = "vulkan")]
    Vulkan(BackendTensor<Vulkan>),

    /// The [WebGPU backend](WebGpu) tensor.
    #[cfg(feature = "webgpu")]
    WebGpu(BackendTensor<WebGpu>),

    /// The [NdArray backend](NdArray) tensor.
    #[cfg(feature = "ndarray")]
    NdArray(BackendTensor<NdArray>),

    /// The [LibTorch backend](LibTorch) tensor.
    #[cfg(feature = "tch")]
    LibTorch(BackendTensor<LibTorch>),
}

impl TensorMetadata for EngineTensor {
    fn dtype(&self) -> burn_std::DType {
        match self {
            #[cfg(feature = "cpu")]
            EngineTensor::Cpu(tensor) => tensor.dtype(),
            #[cfg(feature = "cuda")]
            EngineTensor::Cuda(tensor) => tensor.dtype(),
            #[cfg(feature = "metal")]
            EngineTensor::Metal(tensor) => tensor.dtype(),
            #[cfg(feature = "rocm")]
            EngineTensor::Rocm(tensor) => tensor.dtype(),
            #[cfg(feature = "vulkan")]
            EngineTensor::Vulkan(tensor) => tensor.dtype(),
            #[cfg(feature = "webgpu")]
            EngineTensor::WebGpu(tensor) => tensor.dtype(),
            #[cfg(feature = "ndarray")]
            EngineTensor::NdArray(tensor) => tensor.dtype(),
            #[cfg(feature = "tch")]
            EngineTensor::LibTorch(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> burn_std::Shape {
        match self {
            #[cfg(feature = "cpu")]
            EngineTensor::Cpu(tensor) => tensor.shape(),
            #[cfg(feature = "cuda")]
            EngineTensor::Cuda(tensor) => tensor.shape(),
            #[cfg(feature = "metal")]
            EngineTensor::Metal(tensor) => tensor.shape(),
            #[cfg(feature = "rocm")]
            EngineTensor::Rocm(tensor) => tensor.shape(),
            #[cfg(feature = "vulkan")]
            EngineTensor::Vulkan(tensor) => tensor.shape(),
            #[cfg(feature = "webgpu")]
            EngineTensor::WebGpu(tensor) => tensor.shape(),
            #[cfg(feature = "ndarray")]
            EngineTensor::NdArray(tensor) => tensor.shape(),
            #[cfg(feature = "tch")]
            EngineTensor::LibTorch(tensor) => tensor.shape(),
        }
    }
}

impl QTensorPrimitive for EngineTensor {
    fn scheme(&self) -> &burn_std::QuantScheme {
        match self {
            #[cfg(feature = "cpu")]
            EngineTensor::Cpu(tensor) => tensor.scheme(),
            #[cfg(feature = "cuda")]
            EngineTensor::Cuda(tensor) => tensor.scheme(),
            #[cfg(feature = "metal")]
            EngineTensor::Metal(tensor) => tensor.scheme(),
            #[cfg(feature = "rocm")]
            EngineTensor::Rocm(tensor) => tensor.scheme(),
            #[cfg(feature = "vulkan")]
            EngineTensor::Vulkan(tensor) => tensor.scheme(),
            #[cfg(feature = "webgpu")]
            EngineTensor::WebGpu(tensor) => tensor.scheme(),
            #[cfg(feature = "ndarray")]
            EngineTensor::NdArray(tensor) => tensor.scheme(),
            #[cfg(feature = "tch")]
            EngineTensor::LibTorch(tensor) => tensor.scheme(),
        }
    }
}
