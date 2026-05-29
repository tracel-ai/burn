use crate::backends::*;

#[cfg(feature = "autodiff")]
use burn_backend::ComplexTensor;
use burn_backend::{Backend, BackendTypes, ComplexTensorBackend, DType, Shape, TensorMetadata};
//#[cfg(feature = "complex")]

use crate::CheckpointingStrategy;
#[cfg(feature = "autodiff")]
use alloc::boxed::Box;
#[cfg(feature = "autodiff")]
use burn_backend::tensor::FloatTensor;

// TODO: if we reduce the different associated types for float/int/bool/quantized tensor primitives down to a single
// `B::TensorPrimitive` we can simplify this.

/// Tensor which points to a backend tensor primitive kind.
#[derive(Clone, Debug)]
pub enum BackendTensor<B: BackendTypes> {
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
    #[cfg(feature = "autodiff")]
    /// Autodiff complex tensor handle.
    AutodiffComplex(ComplexTensor<Autodiff<B>>),
    //#[cfg(feature = "complex")]
    /// Complex tensor handle.
    Complex(B::ComplexTensorPrimitive),
}

impl<B: Backend + ComplexTensorBackend> BackendTensor<B> {
    /// Returns the inner float tensor primitive.
    pub fn float(self) -> B::FloatTensorPrimitive {
        match self {
            BackendTensor::Float(tensor) => tensor,
            BackendTensor::Int(_) => panic!("Should be float, got int"),
            BackendTensor::Bool(_) => panic!("Should be float, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be float, got quantized"),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => panic!("Should be float, got autodiff"),
            #[cfg(feature = "autodiff")]
            BackendTensor::AutodiffComplex(_) => panic!("Should be float, got autodiff complex"),
            //#[cfg(feature = "complex")]
            BackendTensor::Complex(_) => panic!("Should be float, got complex"),
        }
    }
    /// Returns the inner float tensor primitive.
    pub fn as_float(&self) -> &B::FloatTensorPrimitive {
        match self {
            BackendTensor::Float(tensor) => tensor,
            BackendTensor::Int(_) => panic!("Should be float, got int"),
            BackendTensor::Bool(_) => panic!("Should be float, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be float, got quantized"),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => panic!("Should be float, got autodiff"),
            #[cfg(feature = "autodiff")]
            BackendTensor::AutodiffComplex(_) => panic!("Should be float, got autodiff complex"),
            //#[cfg(feature = "complex")]
            BackendTensor::Complex(_) => panic!("Should be float, got complex"),
        }
    }

    /// Returns the inner int tensor primitive.
    pub fn int(self) -> B::IntTensorPrimitive {
        match self {
            BackendTensor::Int(tensor) => tensor,
            BackendTensor::Float(_) => panic!("Should be int, got float"),
            BackendTensor::Bool(_) => panic!("Should be int, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be int, got quantized"),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => panic!("Should be int, got autodiff"),
            #[cfg(feature = "autodiff")]
            BackendTensor::AutodiffComplex(_) => panic!("Should be int, got autodiff complex"),
            //#[cfg(feature = "complex")]
            BackendTensor::Complex(_) => panic!("Should be int, got complex"),
        }
    }

    /// Returns the inner bool tensor primitive.
    pub fn bool(self) -> B::BoolTensorPrimitive {
        match self {
            BackendTensor::Bool(tensor) => tensor,
            BackendTensor::Float(_) => panic!("Should be bool, got float"),
            BackendTensor::Int(_) => panic!("Should be bool, got int"),
            BackendTensor::Quantized(_) => panic!("Should be bool, got quantized"),
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => panic!("Should be bool, got autodiff"),
            #[cfg(feature = "autodiff")]
            BackendTensor::AutodiffComplex(_) => panic!("Should be bool, got autodiff complex"),
            //#[cfg(feature = "complex")]
            BackendTensor::Complex(_) => panic!("Should be bool, got complex"),
        }
    }

    /// Returns the inner quantized tensor primitive.
    pub fn quantized(self) -> B::QuantizedTensorPrimitive {
        match self {
            BackendTensor::Quantized(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    ///TODO: Need to figure out how to return the inner autodiff tensor primitive;
    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub fn autodiff_float(self) -> FloatTensor<Autodiff<B>> {
        match self {
            BackendTensor::Autodiff(tensor) => tensor,
            // NOTE: this is the panicking code reached in tensor.rs:74:18:
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub fn autodiff_complex(self) -> ComplexTensor<Autodiff<B>> {
        match self {
            BackendTensor::AutodiffComplex(tensor) => tensor,
            // NOTE: this is the panicking code reached in tensor.rs:74:18:
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub fn as_autodiff_float(&self) -> &FloatTensor<Autodiff<B>> {
        match self {
            BackendTensor::Autodiff(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff complex tensor primitive.
    pub fn as_autodiff_complex(&self) -> &ComplexTensor<Autodiff<B>> {
        match self {
            BackendTensor::AutodiffComplex(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub fn autodiff_inner(self) -> B::FloatTensorPrimitive {
        match self {
            BackendTensor::Autodiff(tensor) => tensor.primitive,
            _ => unreachable!(),
        }
    }
    /// Returns the inner complex tensor primitive.
    pub fn complex(self) -> B::ComplexTensorPrimitive {
        match self {
            BackendTensor::Complex(tensor) => tensor,
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
            #[cfg(feature = "autodiff")]
            BackendTensor::AutodiffComplex(tensor) => {
                // The unimplementedTensorPrimitive needs a placeholder that should never be reached at runtime
                #[allow(unreachable_code)]
                B::complex_device(tensor.primitive())
            }
            //#[cfg(feature = "complex")]
            BackendTensor::Complex(tensor) => B::complex_device(tensor),
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
            #[cfg(feature = "autodiff")]
            BackendTensor::AutodiffComplex(tensor) => tensor.dtype(),
            //#[cfg(feature = "complex")]
            BackendTensor::Complex(tensor) => tensor.dtype(),
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
            #[cfg(feature = "autodiff")]
            BackendTensor::AutodiffComplex(tensor) => tensor.shape(),
            //#[cfg(feature = "complex")]
            BackendTensor::Complex(tensor) => tensor.shape(),
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
    pub kind: DispatchTensorKind,
    // Technically more of a device property, but device is not a dispatch tensor field.
    // Right now this is the easiest way to preserve the checkpointing strategy because primitives are not consolidated.
    // Once float/int/bool primitives are consolidated into a single associative type, we could hold that
    // property for all autodiff tensors.
    /// Holds the autodiff checkpointing strategy.
    /// - `None`: tensor is not tracked by autodiff
    /// - `Some(strategy)`: tensor is tracked by autodiff, and uses the checkpointing `strategy`
    pub checkpointing: Option<CheckpointingStrategy>,
}

impl DispatchTensor {
    /// Returns the tensor kind primitive.
    pub fn into_primitive(self) -> DispatchTensorKind {
        self.kind
    }
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
    #[cfg(feature = "metal")]
    Metal(BackendTensor<Metal>),

    /// The [ROCm backend](Rocm) tensor.
    #[cfg(feature = "rocm")]
    Rocm(BackendTensor<Rocm>),

    /// The [Vulkan backend](Vulkan) tensor.
    #[cfg(feature = "vulkan")]
    Vulkan(BackendTensor<Vulkan>),

    /// The [Wgpu backend](Wgpu) tensor.
    #[cfg(feature = "wgpu")]
    Wgpu(BackendTensor<Wgpu>),

    /// The [WebGPU backend](Wgpu) tensor.
    #[cfg(feature = "webgpu")]
    WebGpu(BackendTensor<WebGpu>),

    /// The [Flex backend](Flex) tensor.
    #[cfg(feature = "flex")]
    Flex(BackendTensor<Flex>),

    /// The [NdArray backend](NdArray) tensor.
    #[cfg(any(feature = "ndarray", default_backend))]
    NdArray(BackendTensor<NdArray>),

    /// The [LibTorch backend](LibTorch) tensor.
    #[cfg(feature = "tch")]
    LibTorch(BackendTensor<LibTorch>),

    /// The [Remote backend](Remote) tensor (lives on a remote server).
    #[cfg(feature = "remote")]
    Remote(BackendTensor<Remote>),

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
            #[cfg(feature = "metal")]
            Self::Metal(tensor) => tensor.dtype(),
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => tensor.dtype(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(tensor) => tensor.dtype(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(tensor) => tensor.dtype(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(tensor) => tensor.dtype(),
            #[cfg(feature = "flex")]
            Self::Flex(tensor) => tensor.dtype(),
            #[cfg(any(feature = "ndarray", default_backend))]
            Self::NdArray(tensor) => tensor.dtype(),
            #[cfg(feature = "tch")]
            Self::LibTorch(tensor) => tensor.dtype(),
            #[cfg(feature = "remote")]
            Self::Remote(tensor) => tensor.dtype(),
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
            #[cfg(feature = "metal")]
            Self::Metal(tensor) => tensor.shape(),
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => tensor.shape(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(tensor) => tensor.shape(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(tensor) => tensor.shape(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(tensor) => tensor.shape(),
            #[cfg(feature = "flex")]
            Self::Flex(tensor) => tensor.shape(),
            #[cfg(any(feature = "ndarray", default_backend))]
            Self::NdArray(tensor) => tensor.shape(),
            #[cfg(feature = "tch")]
            Self::LibTorch(tensor) => tensor.shape(),
            #[cfg(feature = "remote")]
            Self::Remote(tensor) => tensor.shape(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(tensor) => tensor.shape(),
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
