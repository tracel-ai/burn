use alloc::string::String;
pub use burn_common::sync_type::SyncType;

use crate::ops::*;
use crate::tensor::Element;

use super::{BackendBridge, DeviceOps};

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
/// Backend tensor types are all `Clone` + `Send`, which allows them to be safely
/// sent between threads. It is recommended to wrap tensors with [Arc](alloc::sync::Arc),
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
    FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + ActivationOps<Self>
    + Clone
    + Sized
    + Default
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
{
    /// Device type.
    type Device: DeviceOps;

    /// A bridge that can cast tensors to full precision.
    type FullPrecisionBridge: BackendBridge<Self> + 'static;

    /// Tensor primitive to be used for all float operations.
    type FloatTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;
    /// Float element type.
    type FloatElem: Element;

    /// Tensor primitive to be used for all int operations.
    type IntTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;
    /// Int element type.
    type IntElem: Element;

    /// Tensor primitive to be used for all bool operations.
    type BoolTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;

    /// If autodiff is enabled.
    fn ad_enabled() -> bool {
        false
    }

    /// Name of the backend.
    fn name() -> String;

    /// Seed the backend.
    fn seed(seed: u64);

    /// Sync the backend, ensure that all computation are finished.
    fn sync(_device: &Self::Device, _sync_type: SyncType) {}
}

/// Trait that allows a backend to support autodiff.
pub trait AutodiffBackend: Backend {
    /// The inner backend type.
    type InnerBackend: Backend<
        Device = Self::Device,
        FloatElem = Self::FloatElem,
        IntElem = Self::IntElem,
    >;

    /// Gradients type.
    type Gradients: Send;

    /// Backward pass.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor is the last node of computational graph where the gradients are computed.
    ///
    /// # Returns
    ///
    /// The gradients.
    fn backward<const D: usize>(tensor: FloatTensor<Self, D>) -> Self::Gradients;

    /// Returns the gradients of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to extract the gradients from.
    ///
    /// # Returns
    ///
    /// An optional tensor containing the gradient.
    fn grad<const D: usize>(
        tensor: &FloatTensor<Self, D>,
        grads: &Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend, D>>;

    /// Pops the gradients of a tensor and returns them.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to pop the gradients from.
    /// * `grads` - The gradients.
    ///
    /// # Returns
    ///
    /// An optional tensor containing the given gradients.
    fn grad_remove<const D: usize>(
        tensor: &FloatTensor<Self, D>,
        grads: &mut Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend, D>>;

    /// Replace the gradients of a tensor with the one provided.
    ///
    /// If no gradient existed for the provided tensor, register it.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to pop the gradients from.
    /// * `grads` - The gradients.
    /// * `grad` - The updated grad tensor.
    fn grad_replace<const D: usize>(
        tensor: &FloatTensor<Self, D>,
        grads: &mut Self::Gradients,
        grad: FloatTensor<Self::InnerBackend, D>,
    );

    /// Returns the tensor with inner backend type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the inner backend tensor for.
    ///
    /// # Returns
    ///
    /// The inner backend tensor.
    fn inner<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self::InnerBackend, D>;

    /// Returns the tensor with inner backend type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the inner backend tensor for.
    ///
    /// # Returns
    ///
    /// The inner backend tensor.
    fn int_inner<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self::InnerBackend, D>;

    /// Returns the tensor with inner backend type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the inner backend tensor for.
    ///
    /// # Returns
    ///
    /// The inner backend tensor.
    fn bool_inner<const D: usize>(tensor: BoolTensor<Self, D>)
        -> BoolTensor<Self::InnerBackend, D>;

    /// Converts the inner backend tensor to the autodiff backend tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The inner backend tensor to convert.
    ///
    ///
    /// # Returns
    ///
    /// The autodiff backend tensor.
    fn from_inner<const D: usize>(
        tensor: FloatTensor<Self::InnerBackend, D>,
    ) -> FloatTensor<Self, D>;

    /// Converts the inner backend tensor to the autodiff backend tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The inner backend tensor to convert.
    ///
    ///
    /// # Returns
    ///
    /// The autodiff backend tensor.
    fn int_from_inner<const D: usize>(
        tensor: IntTensor<Self::InnerBackend, D>,
    ) -> IntTensor<Self, D>;

    /// Converts the inner backend tensor to the autodiff backend tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The inner backend tensor to convert.
    ///
    ///
    /// # Returns
    ///
    /// The autodiff backend tensor.
    fn bool_from_inner<const D: usize>(
        tensor: BoolTensor<Self::InnerBackend, D>,
    ) -> BoolTensor<Self, D>;
}
