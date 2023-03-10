use alloc::string::String;

use crate::ops::*;
use crate::tensor::Element;

/// This trait defines all types and functions needed for a backend to be used with burn.
///
/// ## Design
///
/// This trait aims to be as unopinionated as possible and allows implementations to define
/// their own types and patterns. Therefore, there are few pre-defined abstractions baked
/// into this trait.
///
/// Backends must define their own tensor types for each data type: `float`, `int`, and `bool`.
/// Since we minimize assumptions, we chose to separate these types, as they are used in
/// different contexts. However, some backends may have a generic tensor type that is used
/// for all data types.
///
/// ### Eager Mode
///
/// Because burn supports dynamic graphs, the backend trait is designed around kernel
/// implementations that can be called without any mutable context or graph. This may not be
/// ideal for backends that want to configure their computational graphs and execute them
/// multiple times.
///
/// To implement this kind of backend, channels could be used to communicate with a backend
/// server thread to build the computation graphs and re-execute the ones that are repeated,
/// with some form of cache. Once that pattern has matured, a graph mode backend trait could
/// be extracted from it, allowing other backends of the same kind to be quickly integrated
/// with burn. This pattern could also be used to create an operation fusion trait, which
/// allows backends to define what kind of graph structures can be fused into one operation.
///
/// ### Multi-Threaded
///
/// Backend tensor types are all `Clone` + `Sync` + `Send`, which allows them to be safely
/// shared between threads. It is recommended to wrap tensors with [Arc](alloc::sync::Arc),
/// which avoids copying the tensor's buffer. Note that it is still possible to mutate and
/// reuse tensors' buffer without locking; see the next section on the Mutable API.
///
/// ### Mutable API
///
/// There is no mutable or inplace operation API to implement, but that does not mean that
/// backends cannot support them. Using [try_unwrap](alloc::sync::Arc::try_unwrap) and
/// [get_mut](alloc::sync::Arc::get_mut) allows backends to have access to an owned or mutable
/// reference to their tensor buffer data structure if the tensor is not shared. In that case,
/// backends can dispatch to their owned inplace operations for better performance.
///
/// ## Documentation
///
/// Most of the documentation for each function can be found on the user API [tensor struct](crate::Tensor).
/// For modules, public functions are often created, which can be used by `burn-core` modules.
pub trait Backend:
    TensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + Clone
    + Sized
    + Default
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
{
    /// Device type.
    type Device: Clone + Default + core::fmt::Debug + Send + Sync;

    /// Pointer to another backend that have a full precision float element type
    type FullPrecisionBackend: Backend<FloatElem = Self::FullPrecisionElem, Device = Self::Device>;
    /// Full precision float element type.
    type FullPrecisionElem: Element;

    /// Tensor primitive to be used for all float operations.
    type TensorPrimitive<const D: usize>: Clone + Send + Sync + 'static + core::fmt::Debug;
    /// Float element type.
    type FloatElem: Element;

    /// Tensor primitive to be used for all int operations.
    type IntTensorPrimitive<const D: usize>: Clone + Send + Sync + 'static + core::fmt::Debug;
    /// Int element type.
    type IntElem: Element + From<i64> + Into<i64>;

    /// Tensor primitive to be used for all bool operations.
    type BoolTensorPrimitive<const D: usize>: Clone + Send + Sync + 'static + core::fmt::Debug;

    /// If autodiff is enabled.
    fn ad_enabled() -> bool {
        false
    }

    /// Name of the backend.
    fn name() -> String;

    /// Seed the backend.
    fn seed(seed: u64);
}

pub(crate) type ADBackendTensorPrimitive<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::TensorPrimitive<D>;

/// Trait that allows a backend to support autodiff.
pub trait ADBackend: Backend {
    type InnerBackend: Backend<Device = Self::Device, FloatElem = Self::FloatElem>;
    type Gradients: Send + Sync;

    fn backward<const D: usize>(tensor: Self::TensorPrimitive<D>) -> Self::Gradients;
    fn grad<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &Self::Gradients,
    ) -> Option<ADBackendTensorPrimitive<D, Self>>;
    fn grad_remove<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &mut Self::Gradients,
    ) -> Option<ADBackendTensorPrimitive<D, Self>>;
    fn inner<const D: usize>(
        tensor: Self::TensorPrimitive<D>,
    ) -> <Self::InnerBackend as Backend>::TensorPrimitive<D>;
    fn from_inner<const D: usize>(
        tensor: <Self::InnerBackend as Backend>::TensorPrimitive<D>,
    ) -> Self::TensorPrimitive<D>;
}
