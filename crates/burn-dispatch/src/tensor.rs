use burn_backend::{
    Backend, DType, QTensorPrimitive, Shape, TensorMetadata, quantization::QuantScheme,
};

#[cfg(feature = "autodiff")]
use crate::CheckpointingStrategy;
use crate::backends::*;

#[cfg(feature = "autodiff")]
use burn_backend::tensor::FloatTensor;

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
    #[cfg(feature = "autodiff")]
    /// Autodiff float tensor handle.
    Autodiff(FloatTensor<Autodiff<B>>),
}

impl<B: Backend> BackendTensor<B> {
    /// Returns the inner float tensor primitive.
    pub(crate) fn float(self) -> B::FloatTensorPrimitive {
        match self {
            BackendTensor::Float(tensor) => tensor,
            BackendTensor::Int(_) => panic!("Should be float, got int"),
            BackendTensor::Bool(_) => panic!("Should be float, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be float, got quantized"),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => panic!("Should be float, got autodiff"),
        }
    }
    /// Returns the inner float tensor primitive.
    pub(crate) fn as_float(&self) -> &B::FloatTensorPrimitive {
        match self {
            BackendTensor::Float(tensor) => tensor,
            BackendTensor::Int(_) => panic!("Should be float, got int"),
            BackendTensor::Bool(_) => panic!("Should be float, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be float, got quantized"),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => panic!("Should be float, got autodiff"),
        }
    }

    /// Returns the inner int tensor primitive.
    pub(crate) fn int(self) -> B::IntTensorPrimitive {
        match self {
            BackendTensor::Int(tensor) => tensor,
            BackendTensor::Float(_) => panic!("Should be int, got float"),
            BackendTensor::Bool(_) => panic!("Should be int, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be int, got quantized"),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => panic!("Should be int, got autodiff"),
        }
    }

    /// Returns the inner bool tensor primitive.
    pub(crate) fn bool(self) -> B::BoolTensorPrimitive {
        match self {
            BackendTensor::Bool(tensor) => tensor,
            BackendTensor::Float(_) => panic!("Should be bool, got float"),
            BackendTensor::Int(_) => panic!("Should be bool, got int"),
            BackendTensor::Quantized(_) => panic!("Should be bool, got quantized"),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => panic!("Should be bool, got autodiff"),
        }
    }

    /// Returns the inner quantized tensor primitive.
    pub(crate) fn quantized(self) -> B::QuantizedTensorPrimitive {
        match self {
            BackendTensor::Quantized(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub(crate) fn autodiff(self) -> FloatTensor<Autodiff<B>> {
        match self {
            BackendTensor::Autodiff(tensor) => tensor,
            // NOTE: this is the panicking code reached in tensor.rs:74:18:
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub(crate) fn as_autodiff(&self) -> &FloatTensor<Autodiff<B>> {
        match self {
            BackendTensor::Autodiff(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub(crate) fn autodiff_inner(self) -> B::FloatTensorPrimitive {
        match self {
            BackendTensor::Autodiff(tensor) => tensor.primitive,
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
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(tensor) => B::float_device(&tensor.primitive),
        }
    }
}

impl<B: Backend> TensorMetadata for BackendTensor<B> {
    fn dtype(&self) -> DType {
        match self {
            BackendTensor::Float(tensor) => tensor.dtype(),
            BackendTensor::Int(tensor) => tensor.dtype(),
            BackendTensor::Bool(tensor) => tensor.dtype(),
            BackendTensor::Quantized(tensor) => tensor.dtype(),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            BackendTensor::Float(tensor) => tensor.shape(),
            BackendTensor::Int(tensor) => tensor.shape(),
            BackendTensor::Bool(tensor) => tensor.shape(),
            BackendTensor::Quantized(tensor) => tensor.shape(),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(tensor) => tensor.shape(),
        }
    }
}

impl<B: Backend> QTensorPrimitive for BackendTensor<B> {
    fn scheme(&self) -> &QuantScheme {
        match self {
            BackendTensor::Quantized(tensor) => tensor.scheme(),
            _ => panic!(
                "Quantization scheme is not valid for dtype {:?}",
                self.dtype(),
            ),
        }
    }
}

/// A tensor that can dispatch operations to any enabled backend at runtime.
///
/// When the `autodiff` feature is enabled, tensors may carry a checkpointing
/// strategy used to control gradient computation. This is derived from the
/// device used to create the tensor.
#[derive(Clone, Debug)]
pub struct DispatchTensor {
    /// Tensor kind primitive.
    pub(crate) kind: DispatchTensorKind,
    // Technically more of a device property, but device is not a dispatch tensor field.
    #[cfg(feature = "autodiff")]
    pub(crate) checkpointing: CheckpointingStrategy,
}

/// Internal representation of a [`DispatchTensor`].
///
/// This enum contains the concrete backend tensor for each enabled backend.
/// It is not intended to be used directly; instead, it is manipulated by
/// the dispatch system to route operations to the correct backend.
///
/// Each variant corresponds to a specific backend implementation.
#[derive(Clone, Debug)]
pub enum DispatchTensorKind {
    /// The [CPU backend](Cpu) tensor.
    #[cfg(feature = "cpu")]
    Cpu(BackendTensor<Cpu>),

    /// The [CUDA backend](Cuda) tensor.
    #[cfg(feature = "cuda")]
    Cuda(BackendTensor<Cuda>),

    /// The [Metal backend](Metal) tensor.
    #[cfg(wgpu_metal)]
    Metal(BackendTensor<Metal>),

    /// The [ROCm backend](Rocm) tensor.
    #[cfg(feature = "rocm")]
    Rocm(BackendTensor<Rocm>),

    /// The [Vulkan backend](Vulkan) tensor.
    #[cfg(wgpu_vulkan)]
    Vulkan(BackendTensor<Vulkan>),

    /// The [WebGPU backend](Wgpu) tensor.
    #[cfg(wgpu_webgpu)]
    Wgpu(BackendTensor<Wgpu>),

    /// The [Flex backend](Flex) tensor.
    #[cfg(feature = "flex")]
    Flex(BackendTensor<Flex>),

    /// The [NdArray backend](NdArray) tensor.
    #[cfg(feature = "ndarray")]
    NdArray(BackendTensor<NdArray>),

    /// The [LibTorch backend](LibTorch) tensor.
    #[cfg(feature = "tch")]
    LibTorch(BackendTensor<LibTorch>),

    /// The [autodiff enabled backend](Autodiff) tensor.
    #[cfg(feature = "autodiff")]
    Autodiff(Box<DispatchTensorKind>),
}

impl TensorMetadata for DispatchTensorKind {
    fn dtype(&self) -> DType {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(tensor) => tensor.dtype(),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => tensor.dtype(),
            #[cfg(wgpu_metal)]
            Self::Metal(tensor) => tensor.dtype(),
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => tensor.dtype(),
            #[cfg(wgpu_vulkan)]
            Self::Vulkan(tensor) => tensor.dtype(),
            #[cfg(wgpu_webgpu)]
            Self::Wgpu(tensor) => tensor.dtype(),
            #[cfg(feature = "flex")]
            Self::Flex(tensor) => tensor.dtype(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(tensor) => tensor.dtype(),
            #[cfg(feature = "tch")]
            Self::LibTorch(tensor) => tensor.dtype(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(tensor) => tensor.shape(),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => tensor.shape(),
            #[cfg(wgpu_metal)]
            Self::Metal(tensor) => tensor.shape(),
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => tensor.shape(),
            #[cfg(wgpu_vulkan)]
            Self::Vulkan(tensor) => tensor.shape(),
            #[cfg(wgpu_webgpu)]
            Self::Wgpu(tensor) => tensor.shape(),
            #[cfg(feature = "flex")]
            Self::Flex(tensor) => tensor.shape(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(tensor) => tensor.shape(),
            #[cfg(feature = "tch")]
            Self::LibTorch(tensor) => tensor.shape(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(tensor) => tensor.shape(),
        }
    }
}

impl QTensorPrimitive for DispatchTensorKind {
    fn scheme(&self) -> &QuantScheme {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(tensor) => tensor.scheme(),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => tensor.scheme(),
            #[cfg(wgpu_metal)]
            Self::Metal(tensor) => tensor.scheme(),
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => tensor.scheme(),
            #[cfg(wgpu_vulkan)]
            Self::Vulkan(tensor) => tensor.scheme(),
            #[cfg(wgpu_webgpu)]
            Self::Wgpu(tensor) => tensor.scheme(),
            #[cfg(feature = "flex")]
            Self::Flex(tensor) => tensor.scheme(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(tensor) => tensor.scheme(),
            #[cfg(feature = "tch")]
            Self::LibTorch(tensor) => tensor.scheme(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(tensor) => tensor.scheme(),
        }
    }
}

impl TensorMetadata for DispatchTensor {
    fn dtype(&self) -> DType {
        self.kind.dtype()
    }

    fn shape(&self) -> Shape {
        self.kind.shape()
    }
}

impl QTensorPrimitive for DispatchTensor {
    fn scheme(&self) -> &QuantScheme {
        self.kind.scheme()
    }
}
