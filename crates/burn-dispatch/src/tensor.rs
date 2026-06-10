use crate::backends::*;

#[cfg(feature = "autodiff")]
use burn_autodiff::checkpoint::strategy::{
    BalancedCheckpointing, CheckpointStrategy, NoCheckpointing,
};
use burn_backend::{Backend, DType, Shape, TensorMetadata};

use crate::CheckpointingStrategy;
#[cfg(feature = "autodiff")]
use alloc::boxed::Box;
#[cfg(feature = "autodiff")]
use burn_backend::tensor::FloatTensor;

use alloc::{format, string::String};

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
    pub fn float(self) -> B::FloatTensorPrimitive {
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
    pub fn as_float(&self) -> &B::FloatTensorPrimitive {
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
    pub fn int(self) -> B::IntTensorPrimitive {
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
    pub fn bool(self) -> B::BoolTensorPrimitive {
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
    pub fn quantized(self) -> B::QuantizedTensorPrimitive {
        match self {
            BackendTensor::Quantized(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub fn autodiff(self) -> FloatTensor<Autodiff<B>> {
        match self {
            BackendTensor::Autodiff(tensor) => tensor,
            // NOTE: this is the panicking code reached in tensor.rs:74:18:
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "autodiff")]
    /// Returns the inner autodiff tensor primitive.
    pub fn as_autodiff(&self) -> &FloatTensor<Autodiff<B>> {
        match self {
            BackendTensor::Autodiff(tensor) => tensor,
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

    /// Returns the tensor primitive kind name.
    pub fn name(&self) -> &'static str {
        match self {
            BackendTensor::Float(_) => "Float",
            BackendTensor::Int(_) => "Int",
            BackendTensor::Bool(_) => "Bool",
            BackendTensor::Quantized(_) => "Quantized",
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => "Autodiff",
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
    #[cfg(any(feature = "flex", default_backend))]
    Flex(BackendTensor<Flex>),

    /// The [NdArray backend](NdArray) tensor.
    #[cfg(feature = "ndarray")]
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
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(tensor) => tensor.dtype(),
            #[cfg(feature = "ndarray")]
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
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(tensor) => tensor.shape(),
            #[cfg(feature = "ndarray")]
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

impl DispatchTensorKind {
    /// Returns the backend tensor kind name.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cpu")]
            DispatchTensorKind::Cpu(_) => "Cpu",
            #[cfg(feature = "cuda")]
            DispatchTensorKind::Cuda(_) => "Cuda",
            #[cfg(feature = "metal")]
            DispatchTensorKind::Metal(_) => "Metal",
            #[cfg(feature = "rocm")]
            DispatchTensorKind::Rocm(_) => "Rocm",
            #[cfg(feature = "vulkan")]
            DispatchTensorKind::Vulkan(_) => "Vulkan",
            #[cfg(feature = "wgpu")]
            DispatchTensorKind::Wgpu(_) => "Wgpu",
            #[cfg(feature = "webgpu")]
            DispatchTensorKind::WebGpu(_) => "WebGpu",
            #[cfg(any(feature = "flex", default_backend))]
            DispatchTensorKind::Flex(_) => "Flex",
            #[cfg(feature = "ndarray")]
            DispatchTensorKind::NdArray(_) => "NdArray",
            #[cfg(feature = "tch")]
            DispatchTensorKind::LibTorch(_) => "LibTorch",
            #[cfg(feature = "remote")]
            DispatchTensorKind::Remote(_) => "Remote",
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(_) => "Autodiff",
        }
    }
}

#[cfg(feature = "autodiff")]
trait IntoCheckpointingStrategy {
    const STRATEGY: CheckpointingStrategy;
}

#[cfg(feature = "autodiff")]
impl IntoCheckpointingStrategy for NoCheckpointing {
    const STRATEGY: CheckpointingStrategy = CheckpointingStrategy::None;
}

#[cfg(feature = "autodiff")]
impl IntoCheckpointingStrategy for BalancedCheckpointing {
    const STRATEGY: CheckpointingStrategy = CheckpointingStrategy::Balanced;
}

/// Trait to execute runtime routing conversions between the dynamic dispatch layer and specific backends.
pub trait DispatchKindConversion<B: Backend> {
    /// Attempts to extract a backend-specific [`BackendTensor`] wrapper from a generic, dynamically-routed [`DispatchTensor`].
    ///
    /// # Errors
    ///
    /// Returns an error if the dynamic routing state does not match the requested backend `B`.
    fn try_into_backend(tensor: DispatchTensor) -> Result<BackendTensor<B>, String>;

    /// Encapsulates a backend-specific tensor variant back into a globally routing [`DispatchTensor`].
    fn from_backend(tensor: BackendTensor<B>) -> DispatchTensor;
}

macro_rules! impl_dispatch_conversion {
    ($backend:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl DispatchKindConversion<$backend> for DispatchTensor {
            fn try_into_backend(tensor: DispatchTensor) -> Result<BackendTensor<$backend>, String> {
                match tensor.kind {
                    DispatchTensorKind::$backend(t) => Ok(t),
                    other => Err(format!(
                        "Expected {} tensor, got variant: {}",
                        stringify!($backend),
                        other.name()
                    )),
                }
            }

            fn from_backend(tensor: BackendTensor<$backend>) -> DispatchTensor {
                DispatchTensor {
                    kind: DispatchTensorKind::$backend(tensor),
                    checkpointing: None,
                }
            }
        }

        #[cfg(all($cfg, feature = "autodiff"))]
        impl<C: CheckpointStrategy + IntoCheckpointingStrategy>
            DispatchKindConversion<Autodiff<$backend, C>> for DispatchTensor
        {
            fn try_into_backend(
                tensor: DispatchTensor,
            ) -> Result<BackendTensor<Autodiff<$backend, C>>, String> {
                match tensor.kind {
                    DispatchTensorKind::Autodiff(t) => match *t {
                        DispatchTensorKind::$backend(t) => match t {
                            // Encode as `BackendTensor::Float` for `Autodiff<B, C>`
                            BackendTensor::Autodiff(t) => Ok(BackendTensor::Float(t)),
                            other => Err(format!(
                                "Expected Autodiff {} float tensor, got Autodiff variant: {}",
                                stringify!($backend),
                                other.name()
                            )),
                        },
                        other => Err(format!(
                            "Expected Autodiff {} tensor, got Autodiff variant: {}",
                            stringify!($backend),
                            other.name()
                        )),
                    },
                    other => Err(format!(
                        "Expected Autodiff tensor, got backend: {}",
                        other.name()
                    )),
                }
            }

            fn from_backend(tensor: BackendTensor<Autodiff<$backend, C>>) -> DispatchTensor {
                // Unwrap the Autodiff backend representation back into the inner hardware representation
                let kind = match tensor {
                    // Inverse: Wrap the `Float` variant back into the backend's `Autodiff` primitive variant
                    BackendTensor::Float(t) => {
                        let ad_tensor = BackendTensor::Autodiff(t);
                        // Wrap in the concrete backend's dispatch container
                        let inner_dispatch = DispatchTensorKind::$backend(ad_tensor);
                        // Re-apply the outer Autodiff dispatch wrapper
                        DispatchTensorKind::Autodiff(Box::new(inner_dispatch))
                    }

                    // Pass-throughs for non-differentiable types
                    BackendTensor::Int(t) => DispatchTensorKind::$backend(BackendTensor::Int(t)),
                    BackendTensor::Bool(t) => DispatchTensorKind::$backend(BackendTensor::Bool(t)),
                    BackendTensor::Quantized(t) => {
                        DispatchTensorKind::$backend(BackendTensor::Quantized(t))
                    }

                    BackendTensor::Autodiff(_) => {
                        panic!("Unexpected Autodiff variant provided to `from_backend`",)
                    }
                };

                DispatchTensor {
                    kind,
                    checkpointing: Some(C::STRATEGY),
                }
            }
        }
    };
}

impl_dispatch_conversion!(Flex, any(feature = "flex", default_backend));
impl_dispatch_conversion!(Cpu, feature = "cpu");
impl_dispatch_conversion!(Cuda, feature = "cuda");
impl_dispatch_conversion!(Rocm, feature = "rocm");
impl_dispatch_conversion!(Remote, feature = "remote");
impl_dispatch_conversion!(Metal, feature = "metal");
impl_dispatch_conversion!(Vulkan, feature = "vulkan");
impl_dispatch_conversion!(Wgpu, feature = "wgpu");
impl_dispatch_conversion!(WebGpu, feature = "webgpu");
impl_dispatch_conversion!(NdArray, feature = "ndarray");
impl_dispatch_conversion!(LibTorch, feature = "tch");
