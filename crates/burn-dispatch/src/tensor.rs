#[cfg(feature = "autodiff")]
use crate::Dispatch;
use crate::{DispatchDevice, backends::*};

#[cfg(feature = "autodiff")]
use burn_autodiff::checkpoint::strategy::{
    BalancedCheckpointing, CheckpointStrategy, NoCheckpointing,
};
use burn_backend::{Backend, BackendTypes, DType, Shape, TensorMetadata};

use crate::GradientCheckpointingStrategy;
#[cfg(feature = "autodiff")]
use alloc::boxed::Box;
#[cfg(feature = "autodiff")]
use burn_backend::{AutodiffBackend, tensor::FloatTensor};

use alloc::{format, string::String};

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
}

impl<B: Backend> BackendTensor<B> {
    /// Returns the inner float tensor primitive.
    pub fn float(self) -> B::FloatTensorPrimitive {
        match self {
            BackendTensor::Float(tensor) => tensor,
            BackendTensor::Int(_) => panic!("Should be float, got int"),
            BackendTensor::Bool(_) => panic!("Should be float, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be float, got quantized"),
        }
    }
    /// Returns the inner float tensor primitive.
    pub fn as_float(&self) -> &B::FloatTensorPrimitive {
        match self {
            BackendTensor::Float(tensor) => tensor,
            BackendTensor::Int(_) => panic!("Should be float, got int"),
            BackendTensor::Bool(_) => panic!("Should be float, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be float, got quantized"),
        }
    }

    /// Returns the inner int tensor primitive.
    pub fn int(self) -> B::IntTensorPrimitive {
        match self {
            BackendTensor::Int(tensor) => tensor,
            BackendTensor::Float(_) => panic!("Should be int, got float"),
            BackendTensor::Bool(_) => panic!("Should be int, got bool"),
            BackendTensor::Quantized(_) => panic!("Should be int, got quantized"),
        }
    }

    /// Returns the inner bool tensor primitive.
    pub fn bool(self) -> B::BoolTensorPrimitive {
        match self {
            BackendTensor::Bool(tensor) => tensor,
            BackendTensor::Float(_) => panic!("Should be bool, got float"),
            BackendTensor::Int(_) => panic!("Should be bool, got int"),
            BackendTensor::Quantized(_) => panic!("Should be bool, got quantized"),
        }
    }

    /// Returns the inner quantized tensor primitive.
    pub fn quantized(self) -> B::QuantizedTensorPrimitive {
        match self {
            BackendTensor::Quantized(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    /// Returns the tensor primitive kind name.
    pub fn name(&self) -> &'static str {
        match self {
            BackendTensor::Float(_) => "Float",
            BackendTensor::Int(_) => "Int",
            BackendTensor::Bool(_) => "Bool",
            BackendTensor::Quantized(_) => "Quantized",
        }
    }
}

impl<B: BackendTypes> TensorMetadata for BackendTensor<B> {
    type Device = B::Device;
    fn device(&self) -> Self::Device {
        match self {
            BackendTensor::Float(tensor) => tensor.device(),
            BackendTensor::Int(tensor) => tensor.device(),
            BackendTensor::Bool(tensor) => tensor.device(),
            BackendTensor::Quantized(tensor) => tensor.device(),
        }
    }
    fn dtype(&self) -> DType {
        match self {
            BackendTensor::Float(tensor) => tensor.dtype(),
            BackendTensor::Int(tensor) => tensor.dtype(),
            BackendTensor::Bool(tensor) => tensor.dtype(),
            BackendTensor::Quantized(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            BackendTensor::Float(tensor) => tensor.shape(),
            BackendTensor::Int(tensor) => tensor.shape(),
            BackendTensor::Bool(tensor) => tensor.shape(),
            BackendTensor::Quantized(tensor) => tensor.shape(),
        }
    }

    fn can_mut(&self) -> bool {
        match self {
            BackendTensor::Float(tensor) => tensor.can_mut(),
            BackendTensor::Int(tensor) => tensor.can_mut(),
            BackendTensor::Bool(tensor) => tensor.can_mut(),
            BackendTensor::Quantized(tensor) => tensor.can_mut(),
        }
    }
}

/// The autodiff backend associated with a dispatch tensor.
///
/// This is backend metadata, not a statement that a float tensor requires
/// gradients. Enabled float tensors may be tracked or untracked by autodiff.
///
/// Tensor inputs to one backend operation merge their autodiff contexts. A disabled context is
/// compatible with an enabled one and is treated as a constant for that operation. Two enabled
/// contexts must use the same gradient-checkpointing strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DispatchAutodiffContext {
    /// Route the tensor through its concrete backend.
    #[default]
    Disabled,
    /// Associate the tensor with an autodiff backend using the given strategy.
    Enabled(GradientCheckpointingStrategy),
}

impl DispatchAutodiffContext {
    /// Merge two tensor contexts into the context used to execute an operation and wrap its output.
    ///
    /// Disabled tensors don't acquire a new context themselves. When combined with an enabled
    /// tensor, they are treated as constants while the operation and its output use the enabled
    /// context.
    ///
    /// # Panics
    ///
    /// Panics when both contexts are enabled with different gradient-checkpointing strategies.
    #[doc(hidden)]
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Disabled, context) | (context, Self::Disabled) => context,
            (Self::Enabled(lhs), Self::Enabled(rhs)) => {
                assert_eq!(
                    lhs, rhs,
                    "Gradient checkpointing strategy mismatch: {lhs:?} vs {rhs:?}. Tensors in the same operation must share a strategy."
                );
                Self::Enabled(lhs)
            }
        }
    }
}

/// A tensor that can dispatch operations to any enabled backend at runtime.
#[derive(Clone, Debug)]
pub struct DispatchTensor {
    /// Tensor kind primitive.
    pub kind: DispatchTensorKind,
    /// Autodiff backend association for this tensor.
    pub autodiff: DispatchAutodiffContext,
}

/// Internal representation of a [`DispatchTensor`].
///
/// This enum contains the concrete backend tensor for each enabled backend.
/// It is not intended to be used directly; instead, it is manipulated by
/// the dispatch system to route operations to the correct backend.
///
/// Each variant corresponds to a specific backend implementation.
#[derive(Clone, Debug)]
// With autodiff and a lone Cube or Remote backend, the only other variant is Autodiff's 8-byte
// box, so the lint fires. Boxing backend tensors would cost every tensor an allocation.
#[allow(clippy::large_enum_variant)]
pub enum DispatchTensorKind {
    #[cfg(not(backend_enabled))]
    #[doc(hidden)]
    Unavailable(crate::NoBackend),
    /// A tensor on the [cubecl backend](Cube) — its device says which runtime.
    #[cfg(cube_backend)]
    Cube(BackendTensor<Cube>),

    /// The [Flex backend](Flex) tensor.
    #[cfg(feature = "flex")]
    Flex(BackendTensor<Flex>),

    /// The [Remote backend](Remote) tensor (lives on a remote server).
    #[cfg(feature = "remote")]
    Remote(BackendTensor<Remote>),
    /// A tensor recorded by the capture backend.
    #[cfg(feature = "capture")]
    Capture(BackendTensor<Capture>),
    /// A tensor split over a [sharded compute device](ShardedCompute).
    #[cfg(feature = "shard")]
    ShardedCompute(BackendTensor<ShardedCompute>),

    /// The [autodiff enabled backend](Autodiff) tensor.
    ///
    /// The primitive is itself a [`DispatchTensor`], so levels nest in the data: a tracked
    /// tensor's primitive may be another tracked tensor.
    #[cfg(feature = "autodiff")]
    Autodiff(Box<FloatTensor<Autodiff<Dispatch>>>),
}

impl TensorMetadata for DispatchTensorKind {
    type Device = DispatchDevice;

    fn dtype(&self) -> DType {
        match self {
            #[cfg(not(backend_enabled))]
            Self::Unavailable(never) => never.unreachable(),
            #[cfg(cube_backend)]
            Self::Cube(tensor) => tensor.dtype(),
            #[cfg(feature = "flex")]
            Self::Flex(tensor) => tensor.dtype(),
            #[cfg(feature = "remote")]
            Self::Remote(tensor) => tensor.dtype(),
            #[cfg(feature = "shard")]
            Self::ShardedCompute(tensor) => tensor.dtype(),
            #[cfg(feature = "capture")]
            Self::Capture(tensor) => tensor.dtype(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            #[cfg(not(backend_enabled))]
            Self::Unavailable(never) => never.unreachable(),
            #[cfg(cube_backend)]
            Self::Cube(tensor) => tensor.shape(),
            #[cfg(feature = "flex")]
            Self::Flex(tensor) => tensor.shape(),
            #[cfg(feature = "remote")]
            Self::Remote(tensor) => tensor.shape(),
            #[cfg(feature = "shard")]
            Self::ShardedCompute(tensor) => tensor.shape(),
            #[cfg(feature = "capture")]
            Self::Capture(tensor) => tensor.shape(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(tensor) => tensor.shape(),
        }
    }

    fn device(&self) -> DispatchDevice {
        match self {
            #[cfg(not(backend_enabled))]
            Self::Unavailable(never) => never.unreachable(),
            #[cfg(cube_backend)]
            DispatchTensorKind::Cube(tensor) => DispatchDevice::Cube(tensor.device()),
            #[cfg(feature = "flex")]
            DispatchTensorKind::Flex(tensor) => DispatchDevice::Flex(tensor.device()),
            #[cfg(feature = "remote")]
            DispatchTensorKind::Remote(tensor) => DispatchDevice::Remote(tensor.device()),
            #[cfg(feature = "shard")]
            DispatchTensorKind::ShardedCompute(tensor) => {
                DispatchDevice::ShardedCompute(tensor.device())
            }
            #[cfg(feature = "capture")]
            DispatchTensorKind::Capture(tensor) => DispatchDevice::Capture(tensor.device()),
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(tensor) => DispatchDevice::autodiff(tensor.device()),
        }
    }

    fn can_mut(&self) -> bool {
        match self {
            #[cfg(not(backend_enabled))]
            Self::Unavailable(never) => never.unreachable(),
            #[cfg(cube_backend)]
            Self::Cube(tensor) => tensor.can_mut(),
            #[cfg(feature = "flex")]
            Self::Flex(tensor) => tensor.can_mut(),
            #[cfg(feature = "remote")]
            Self::Remote(tensor) => tensor.can_mut(),
            #[cfg(feature = "shard")]
            Self::ShardedCompute(tensor) => tensor.can_mut(),
            #[cfg(feature = "capture")]
            Self::Capture(tensor) => tensor.can_mut(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(tensor) => tensor.can_mut(),
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

    fn can_mut(&self) -> bool {
        self.kind.can_mut()
    }

    type Device = DispatchDevice;

    fn device(&self) -> Self::Device {
        #[allow(unused_mut)]
        let mut device = self.kind.device();

        #[cfg(feature = "autodiff")]
        match (&self.kind, self.autodiff) {
            (DispatchTensorKind::Autodiff(_), DispatchAutodiffContext::Disabled) => {
                panic!("an autodiff float primitive must have an enabled autodiff context")
            }
            (DispatchTensorKind::Autodiff(_), DispatchAutodiffContext::Enabled(strategy)) => {
                let DispatchDevice::Autodiff(device) = &mut device else {
                    unreachable!("autodiff primitive must report an autodiff device")
                };
                device.checkpointing = strategy;
            }
            (_, DispatchAutodiffContext::Enabled(strategy)) => {
                if self.dtype().is_float() {
                    panic!("an enabled float tensor must use an autodiff primitive")
                }
                device = DispatchDevice::autodiff(device);
                let DispatchDevice::Autodiff(device) = &mut device else {
                    unreachable!()
                };
                device.checkpointing = strategy;
            }
            (_, DispatchAutodiffContext::Disabled) => {}
        }

        device
    }
}

#[cfg(feature = "shard")]
impl DispatchTensorKind {
    /// Borrows the kind without any autodiff level.
    pub(crate) fn concrete(&self) -> &Self {
        match self {
            #[cfg(feature = "autodiff")]
            Self::Autodiff(tensor) => tensor.primitive().kind.concrete(),
            kind => kind,
        }
    }
}

#[cfg(feature = "autodiff")]
impl DispatchTensor {
    /// Converts an extension field into the backend tensor of its autodiff level.
    ///
    /// A tracked float is unwrapped, a concrete float starts a new leaf on `AD`, and int, bool and
    /// quantized tensors pass through with a disabled context: `Autodiff<Dispatch>` has no level for
    /// them. Extension fields carry their own contexts, so the conversion is per field rather than
    /// per tensor.
    #[doc(hidden)]
    pub fn into_autodiff_backend<AD>(self) -> BackendTensor<AD>
    where
        AD: AutodiffBackend<
                InnerBackend = Dispatch,
                FloatTensorPrimitive = FloatTensor<Autodiff<Dispatch>>,
                IntTensorPrimitive = DispatchTensor,
                BoolTensorPrimitive = DispatchTensor,
                QuantizedTensorPrimitive = DispatchTensor,
            >,
    {
        let context = self.autodiff;
        match self.kind {
            DispatchTensorKind::Autodiff(tensor) => {
                assert!(
                    matches!(context, DispatchAutodiffContext::Enabled(_)),
                    "an autodiff float primitive must have an enabled autodiff context"
                );
                BackendTensor::Float(*tensor)
            }
            kind => {
                let tensor = DispatchTensor {
                    kind,
                    autodiff: DispatchAutodiffContext::Disabled,
                };
                match tensor.dtype() {
                    DType::Bool(_) => BackendTensor::Bool(tensor),
                    DType::QFloat(_) => BackendTensor::Quantized(tensor),
                    dtype if dtype.is_float() => {
                        assert_eq!(
                            context,
                            DispatchAutodiffContext::Disabled,
                            "an enabled float tensor must use an autodiff primitive"
                        );
                        BackendTensor::Float(<AD as AutodiffBackend>::from_inner(tensor))
                    }
                    _ => BackendTensor::Int(tensor),
                }
            }
        }
    }
}

impl DispatchTensor {
    /// Consumes the tensor and strips every autodiff level, returning its value.
    pub(crate) fn into_concrete(self) -> Self {
        let DispatchTensor { kind, .. } = self;
        match kind {
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(tensor) => tensor.into_primitive().into_concrete(),
            kind => DispatchTensor {
                kind,
                autodiff: DispatchAutodiffContext::Disabled,
            },
        }
    }
}

#[cfg(feature = "autodiff")]
impl DispatchTensorKind {
    /// Wraps a tracked tensor as a dispatch kind.
    pub fn autodiff(tensor: FloatTensor<Autodiff<Dispatch>>) -> Self {
        Self::Autodiff(Box::new(tensor))
    }
}

impl DispatchTensorKind {
    /// Returns the backend tensor kind name.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            #[cfg(not(backend_enabled))]
            Self::Unavailable(never) => never.unreachable(),
            #[cfg(cube_backend)]
            DispatchTensorKind::Cube(_) => "Cube",
            #[cfg(feature = "flex")]
            DispatchTensorKind::Flex(_) => "Flex",
            #[cfg(feature = "remote")]
            DispatchTensorKind::Remote(_) => "Remote",
            #[cfg(feature = "shard")]
            DispatchTensorKind::ShardedCompute(_) => "ShardedCompute",
            #[cfg(feature = "capture")]
            DispatchTensorKind::Capture(_) => "Capture",
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(_) => "Autodiff",
        }
    }
}

#[cfg(feature = "autodiff")]
trait IntoGradientCheckpointingStrategy {
    const STRATEGY: GradientCheckpointingStrategy;
}

#[cfg(feature = "autodiff")]
impl IntoGradientCheckpointingStrategy for NoCheckpointing {
    const STRATEGY: GradientCheckpointingStrategy = GradientCheckpointingStrategy::Disabled;
}

#[cfg(feature = "autodiff")]
impl IntoGradientCheckpointingStrategy for BalancedCheckpointing {
    const STRATEGY: GradientCheckpointingStrategy = GradientCheckpointingStrategy::Balanced;
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
                if tensor.autodiff != DispatchAutodiffContext::Disabled {
                    return Err(format!(
                        "Expected concrete {} backend with disabled autodiff context, got {:?}",
                        stringify!($backend),
                        tensor.autodiff,
                    ));
                }
                // The catch-all is unreachable in single-backend builds (the enum then has one
                // variant), but required when several backend features are enabled.
                #[allow(unreachable_patterns)]
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
                    autodiff: DispatchAutodiffContext::Disabled,
                }
            }
        }
    };
}

#[cfg(feature = "autodiff")]
impl<C: CheckpointStrategy + IntoGradientCheckpointingStrategy>
    DispatchKindConversion<Autodiff<Dispatch, C>> for DispatchTensor
{
    fn try_into_backend(
        tensor: DispatchTensor,
    ) -> Result<BackendTensor<Autodiff<Dispatch, C>>, String> {
        if tensor.autodiff != DispatchAutodiffContext::Enabled(C::STRATEGY) {
            return Err(format!(
                "Expected {:?} autodiff context, got {:?}",
                C::STRATEGY,
                tensor.autodiff
            ));
        }
        let DispatchTensor { kind, .. } = tensor;
        match kind {
            DispatchTensorKind::Autodiff(tensor) => Ok(BackendTensor::Float(*tensor)),
            other => {
                let dtype = other.dtype();
                let tensor = DispatchTensor {
                    kind: other,
                    autodiff: DispatchAutodiffContext::Disabled,
                };
                match dtype {
                    DType::Bool(_) => Ok(BackendTensor::Bool(tensor)),
                    DType::QFloat(_) => Ok(BackendTensor::Quantized(tensor)),
                    dtype if dtype.is_float() => Err(format!(
                        "Expected Autodiff float tensor, got an untracked {dtype:?} tensor"
                    )),
                    _ => Ok(BackendTensor::Int(tensor)),
                }
            }
        }
    }

    fn from_backend(tensor: BackendTensor<Autodiff<Dispatch, C>>) -> DispatchTensor {
        let DispatchTensor { kind, .. } = match tensor {
            BackendTensor::Float(tensor) => DispatchTensor {
                kind: DispatchTensorKind::Autodiff(Box::new(tensor)),
                autodiff: DispatchAutodiffContext::Disabled,
            },
            BackendTensor::Int(tensor)
            | BackendTensor::Bool(tensor)
            | BackendTensor::Quantized(tensor) => tensor,
        };
        DispatchTensor {
            kind,
            autodiff: DispatchAutodiffContext::Enabled(C::STRATEGY),
        }
    }
}

// One invocation per dispatch variant. Every cubecl runtime is the same `Cube`
// backend, so they share the one impl rather than getting seven identical ones.
impl_dispatch_conversion!(Cube, cube_backend);
impl_dispatch_conversion!(Flex, feature = "flex");
impl_dispatch_conversion!(Remote, feature = "remote");
impl_dispatch_conversion!(ShardedCompute, feature = "shard");
impl_dispatch_conversion!(Capture, feature = "capture");
