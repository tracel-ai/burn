use alloc::string::String;

use crate::ops::*;
use crate::tensor::Element;

/// Trait defining all types and functions needed for a backend to be used with all of burn.
///
/// ## Design
///
/// This trait tries to be the less opiniontated possible and let implementations define
/// their own types and patterns. This is why there is almost no pre-defined abstractions
/// baked into this trait.
///
/// Backends have to define their own tensor types for each data type: float, int and bool. Since
/// we minimize assuptions, the choise has been made to separate those types, since they are used
/// in different contexts. Some backends may have a generic tensor type that is used for all data
/// types, but this is not assumed.
///
/// ### Eager
///
/// Since burn supports dynamical graphs, the backend trait is designed over kernel implementations that
/// can be called without any mutable context or graph. This may not be ideal for backends that
/// want to configure their computational graphs and execute them multiple times.
///
/// To implement that kind of backend, channels could be used to communicate with a backend server
/// thread to build the computation graphs and re-execute the ones that are repeated with some form of
/// cache. Once that pattern has matured, a graph mode backend trait could be extracted from it
/// allowing other backend of the same kind to quickly be integrated with burn.
/// This pattern could also be used to create an operation fusion trait which let backends defined
/// what kind of graph structures can be fused into one operation.
///
/// ### Multi Thread
///
/// Backend tensor types are all `Clone` + `Sync` + `Send`, which allows them to safely be shared
/// between threads. Using [Arc](alloc::sync::Arc) is recommended to wrap tensor, which avoid
/// copying tensor's buffer. Note that it's still possible to mutate and reuse tensors' buffer
/// without locking, see the next section on Mutable API.
///
/// ### Mutable API
///
/// There is no mutable or inplace operation API to implement, but it doesn't mean that backend
/// should not support them. Using [try_unwrap](alloc::sync::Arc::try_unwrap) and
/// [get_mut](alloc::sync::Arc::get_mut) allow backend to have access to a owned or mutable
/// reference to their tensor buffer data strcuture if the tensor is not shared. In that case,
/// backends can dispatch to their owned inplace operations for better performance.
///
/// Note that this also allows other kind of optimizations. Here's some example from the tch
/// backend:
///
/// ```
///fn lower<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<bool, D> {
///    let _tensor = TchTensor::binary_ops_tensor(
///        lhs,
///        rhs,
///        // Lhs is not shared, therefore safe to mutate.
///        |lhs, rhs| lhs.less_tensor_(rhs).to_kind(tch::Kind::Bool),
///        // Lhs is shared, therefore not safe to mutate, but rhs is not shared.
///        |lhs, rhs| rhs.greater_tensor_(lhs).to_kind(tch::Kind::Bool),
///        /// Lhs and Rhs are shared, creating a new output buffer.
///        |lhs, rhs| lhs.less_tensor(rhs),
///    );
///    todo!();
///}
/// ```
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

    fn ad_enabled() -> bool;
    fn name() -> String;
    fn seed(seed: u64);
}

pub(crate) type ADBackendTensorPrimitive<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::TensorPrimitive<D>;

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
