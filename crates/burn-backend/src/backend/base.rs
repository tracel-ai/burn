use burn_std::DType;
pub use burn_std::backtrace::BackTrace;

use alloc::string::String;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::element::Element;
use crate::ops::*;
use crate::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use crate::{QTensorPrimitive, TensorData, TensorMetadata};

use super::DeviceOps;

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
/// Most of the documentation for each function can be found on the user API
#[cfg_attr(doc, doc = crate::doc_tensor!())]
#[cfg_attr(not(doc), doc = "`Tensor`")]
/// struct in the `burn-tensor` crate.
/// For modules, public functions are often created, which can be used by `burn-core` modules.
pub trait Backend:
    FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + ActivationOps<Self>
    + QTensorOps<Self>
    + TransactionOps<Self>
    + Clone
    + Default
    + Sized
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
{
    /// Device type.
    type Device: DeviceOps;

    /// Tensor primitive to be used for all float operations.
    type FloatTensorPrimitive: TensorMetadata + 'static;
    /// Default float element type.
    type FloatElem: Element;

    /// Tensor primitive to be used for all int operations.
    type IntTensorPrimitive: TensorMetadata + 'static;
    /// Int element type.
    type IntElem: Element;

    /// Tensor primitive to be used for all bool operations.
    type BoolTensorPrimitive: TensorMetadata + 'static;
    /// Tensor primitive to be used for all bool operations.
    type BoolElem: Element;

    /// Tensor primitive to be used for all quantized operations.
    type QuantizedTensorPrimitive: TensorMetadata + QTensorPrimitive + 'static;

    /// If autodiff is enabled.
    fn ad_enabled() -> bool {
        false
    }

    /// Sets the current allocation mode to persistent.
    #[allow(unused_variables)]
    fn memory_persistent_allocations<Output, Input, Func: Fn(Input) -> Output>(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        func(input)
    }

    /// Manually triggers a memory cleanup on the given device.
    #[allow(unused_variables)]
    fn memory_cleanup(device: &Self::Device) {}

    /// Name of the backend.
    fn name(device: &Self::Device) -> String;

    /// Seeds the backend on the specified device.
    ///
    /// There is no guarantee that only the specified device will be seeded, but it is guaranteed
    /// that at least the specified device will be seeded.
    ///
    /// In all cases, this should ensure deterministic execution for a single-threaded program.
    fn seed(device: &Self::Device, seed: u64);

    /// Sync the backend, ensure that all computation are finished.
    fn sync(_device: &Self::Device) -> Result<(), ExecutionError> {
        Ok(())
    }

    /// Marks the given data as being used as a staging buffer for transfer between CPU and
    /// accelerators like GPUs.
    ///
    /// The given data might be transferred to pinned memory or another format to improve data transfer
    /// speed.
    fn staging<'a, Iter>(_data: Iter, _device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
    }

    /// Whether the type is supported by the specified device.
    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool;
}

/// An error that can happen when syncing a device.
#[derive(Error, Serialize, Deserialize)]
pub enum ExecutionError {
    /// A generic error happened during execution.
    ///
    /// The backtrace and context information should be included in the reason string.
    #[error("An error happened during execution\nCaused by:\n  {reason}")]
    WithContext {
        /// The reason of the error.
        reason: String,
    },
    /// A generic error happened during execution thrown in the Burn project.
    ///
    /// The full context isn't captured by the string alone.
    #[error("An error happened during execution\nCaused by:\n  {reason}")]
    Generic {
        /// The reason of the error.
        reason: String,
        /// The backtrace.
        #[serde(skip)]
        backtrace: BackTrace,
    },
}

impl core::fmt::Debug for ExecutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Trait that allows a backend to support autodiff.
pub trait AutodiffBackend: Backend {
    /// The inner backend type.
    type InnerBackend: Backend<Device = Self::Device, FloatElem = Self::FloatElem, IntElem = Self::IntElem>;

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
    fn backward(tensor: FloatTensor<Self>) -> Self::Gradients;

    /// Returns the gradients of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to extract the gradients from.
    ///
    /// # Returns
    ///
    /// An optional tensor containing the gradient.
    fn grad(
        tensor: &FloatTensor<Self>,
        grads: &Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend>>;

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
    fn grad_remove(
        tensor: &FloatTensor<Self>,
        grads: &mut Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend>>;

    /// Replace the gradients of a tensor with the one provided.
    ///
    /// If no gradient existed for the provided tensor, register it.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to pop the gradients from.
    /// * `grads` - The gradients.
    /// * `grad` - The updated grad tensor.
    fn grad_replace(
        tensor: &FloatTensor<Self>,
        grads: &mut Self::Gradients,
        grad: FloatTensor<Self::InnerBackend>,
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
    fn inner(tensor: FloatTensor<Self>) -> FloatTensor<Self::InnerBackend>;

    /// Returns the tensor with inner backend type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the inner backend tensor for.
    ///
    /// # Returns
    ///
    /// The inner backend tensor.
    fn int_inner(tensor: IntTensor<Self>) -> IntTensor<Self::InnerBackend>;

    /// Returns the tensor with inner backend type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the inner backend tensor for.
    ///
    /// # Returns
    ///
    /// The inner backend tensor.
    fn bool_inner(tensor: BoolTensor<Self>) -> BoolTensor<Self::InnerBackend>;

    /// Returns the tensor with inner backend type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the inner backend tensor for.
    ///
    /// # Returns
    ///
    /// The inner backend tensor.
    fn q_inner(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self::InnerBackend>;

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
    fn from_inner(tensor: FloatTensor<Self::InnerBackend>) -> FloatTensor<Self>;

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
    fn int_from_inner(tensor: IntTensor<Self::InnerBackend>) -> IntTensor<Self>;

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
    fn bool_from_inner(tensor: BoolTensor<Self::InnerBackend>) -> BoolTensor<Self>;

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
    fn q_from_inner(tensor: QuantizedTensor<Self::InnerBackend>) -> QuantizedTensor<Self>;
}
