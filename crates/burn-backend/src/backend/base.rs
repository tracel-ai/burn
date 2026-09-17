use burn_std::DType;
pub use burn_std::{ExecutionError, backtrace::BackTrace};

use crate::distributed::DistributedOps;
pub use crate::element::Element;
use crate::ops::*;
use crate::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use crate::{TensorData, TensorMetadata};
use alloc::string::String;
use alloc::vec::Vec;
use enumset::{EnumSet, EnumSetType};

use crate::distributed::{DistributedParamId, DistributedParams};

use super::DeviceOps;
use super::{InstallMemoryPoolsError, MemoryPoolLayout, MemoryPoolUsage, SlicedPoolReport};
use super::{ProfileDuration, ProfileOptions, ProfileToken, profile_system_time};

/// The mapping of types used by Backend and traits.
pub trait BackendTypes: Clone + Send + Sync + core::fmt::Debug + 'static {
    /// Device type.
    type Device: DeviceOps;

    /// Tensor primitive to be used for all float operations.
    type FloatTensorPrimitive: TensorMetadata<Device = Self::Device> + 'static;

    /// Tensor primitive to be used for all int operations.
    type IntTensorPrimitive: TensorMetadata<Device = Self::Device> + 'static;

    /// Tensor primitive to be used for all bool operations.
    type BoolTensorPrimitive: TensorMetadata<Device = Self::Device> + 'static;

    /// Tensor primitive to be used for all quantized operations.
    type QuantizedTensorPrimitive: TensorMetadata<Device = Self::Device> + 'static;

    /// Captured graph primitive returned by [`Backend::graph_stop_capture`] and
    /// consumed by [`Backend::graph_replay`]: a backend-owned recording of a
    /// launch sequence that replays as a single dispatch.
    ///
    /// Backends without graph-capture support use [`GraphUnsupported`], an
    /// uninhabited type — their capture methods only ever error, so no value of
    /// it can exist.
    type GraphPrimitive: Clone + Send + Sync + core::fmt::Debug + 'static;
}

/// Captured graph primitive type used by the backend (see
/// [`BackendTypes::GraphPrimitive`]).
pub type BackendGraph<B> = <B as BackendTypes>::GraphPrimitive;

/// Placeholder [graph primitive](BackendTypes::GraphPrimitive) for backends
/// without graph-capture support.
///
/// Uninhabited: `graph_stop_capture` on such backends always errors, so a value
/// of this type can never be constructed (and `graph_replay` can never be called).
#[derive(Debug, Clone, Copy)]
pub enum GraphUnsupported {}

/// The error returned by the default (unsupported) graph-capture methods.
fn graph_unsupported() -> ExecutionError {
    ExecutionError::Generic {
        reason: alloc::string::String::from("graph capture is not supported by this backend"),
        backtrace: BackTrace::capture(),
    }
}

fn profile_unsupported() -> ExecutionError {
    ExecutionError::Generic {
        reason: alloc::string::String::from(
            "profiling windows are not supported by this backend; use `profile`",
        ),
        backtrace: BackTrace::capture(),
    }
}

/// A closure-bracketed window over a backend's split
/// [`profile_start`](Backend::profile_start) / [`profile_end`](Backend::profile_end),
/// falling back to [`profile_system_time`] when the backend opens none.
///
/// What a forwarding backend measures with, when what it forwards to may or
/// may not have a device clock, and what a backend with a device clock
/// measures with when the closure must not run under a hold of the device.
pub fn profile_with_tokens<B: Backend, O: Send + 'static>(
    device: &B::Device,
    options: ProfileOptions,
    func: impl FnOnce() -> O + Send,
) -> Result<(O, ProfileDuration), ExecutionError> {
    let opened = match B::profile_start(device) {
        Ok(Some(token)) => token,
        Ok(None) => return profile_system_time::<B, O>(device, func),
        // The window could not be opened — a remote device reached from a
        // browser thread, which cannot wait on the server. `func` still runs:
        // a caller asked for their work to be measured, and handing back an
        // error having quietly skipped the work is the one outcome they
        // cannot recover from. Measuring is what failed, so only the
        // measurement is lost.
        Err(err) => {
            func();
            return Err(err);
        }
    };

    // Held so an unwinding `func` abandons the window instead of leaving it
    // open on the server for the rest of the process.
    let mut window = OpenWindow::<B> {
        device,
        token: Some(opened),
    };
    let out = func();
    let token = window
        .token
        .take()
        .expect("the window is held from opening until here");
    let duration = B::profile_end(device, token, options)?;
    Ok((out, duration))
}

/// An open profiling window, abandoned on drop unless it was taken to be
/// closed. See [`Backend::profile_abandon`].
struct OpenWindow<'a, B: Backend> {
    device: &'a B::Device,
    token: Option<ProfileToken>,
}

impl<B: Backend> Drop for OpenWindow<'_, B> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            B::profile_abandon(self.device, token);
        }
    }
}

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
    BackendTypes
    + FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + ActivationOps<Self>
    + QTensorOps<Self>
    + TransactionOps<Self>
    + DistributedOps<Self>
    + Clone
    + Default
    + Sized
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
{
    /// If autodiff is enabled.
    fn ad_enabled(_device: &Self::Device) -> bool {
        false
    }

    /// Sets the current allocation mode to persistent.
    #[allow(unused_variables)]
    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        func(input)
    }

    /// Manually triggers a memory cleanup on the given device.
    #[allow(unused_variables)]
    fn memory_cleanup(device: &Self::Device) {}

    /// Install a layout for the device's dynamic memory pools.
    ///
    /// A per-workload setting: the calling stream's pools are rebuilt in place
    /// when nothing is live in them — so install at a quiescent point, after
    /// the previous workload's tensors have dropped and a
    /// [`memory_cleanup`](Self::memory_cleanup) — and streams created
    /// afterwards use the new layout.
    ///
    /// Sizing a layout from a measurement means installing twice: once
    /// growable, to run the workload and read
    /// [`memory_pool_report`](Self::memory_pool_report), and once capped at
    /// what that reported.
    ///
    /// # Errors
    ///
    /// [`InstallMemoryPoolsError::PoolsInUse`] when something is still live in
    /// the pools being rebuilt, worth retrying once it drains;
    /// [`InvalidLayout`](InstallMemoryPoolsError::InvalidLayout) when the
    /// layout cannot be honoured; and
    /// [`Unsupported`](InstallMemoryPoolsError::Unsupported) — the default — on
    /// a backend with no configurable pools. Neither of the last two is worth a
    /// retry. The layout in force is unchanged in every case, so a caller that
    /// cannot proceed without it has to say so rather than assume the
    /// reservation it asked for.
    #[allow(unused_variables)]
    fn memory_install_pools(
        device: &Self::Device,
        layout: MemoryPoolLayout,
    ) -> Result<(), InstallMemoryPoolsError> {
        Err(InstallMemoryPoolsError::Unsupported)
    }

    /// The dynamic pools' measured state, in the order allocations are routed
    /// through them. `None` on a backend that does not report one, or whose
    /// stream has failed.
    ///
    /// Entries pair one-to-one with the pools of a
    /// [`Sliced`](MemoryPoolLayout::Sliced) or
    /// [`Direct`](MemoryPoolLayout::Direct) layout this caller installed, which
    /// is what a measured layout is rebuilt from. A layout nobody installed —
    /// the runtime's default, or a preset — also routes through pools of other
    /// kinds, which are left out, so its entries carry no rebuildable position.
    ///
    /// Reporting and installing are separate capabilities: a runtime may
    /// describe the pools it has while refusing to be given different ones, so
    /// a report is not proof that a layout was installed. Only the result of
    /// [`memory_install_pools`](Self::memory_install_pools) says that.
    #[allow(unused_variables)]
    fn memory_pool_report(device: &Self::Device) -> Option<Vec<SlicedPoolReport>> {
        None
    }

    /// The device allocator's current state. `None` on a backend that does not
    /// report one, or whose stream has failed.
    #[allow(unused_variables)]
    fn memory_pool_usage(device: &Self::Device) -> Option<MemoryPoolUsage> {
        None
    }

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

    /// Measure how long the device spends on the work `func` puts on the
    /// calling stream, in device time.
    ///
    /// The window opens where the stream is when the call is made and closes
    /// where the stream is when `func` returns: work the stream still owed
    /// from before falls in, and work a backend queues past the end (a
    /// batching backend's last operations, unless `options` flush) falls out.
    /// Nothing is waited on — the [`ProfileDuration`] resolves later, when
    /// the device has stamped both ends — so windows nest without the inner
    /// ones being charged to the outer. Work on other streams is not kept
    /// out, and not counted. A window that nothing ran in reads as no time.
    ///
    /// `name` labels the window for a tracing profiler, on a backend whose
    /// window carries one.
    ///
    /// The default is [`profile_system_time`]: wall-clock time between two
    /// syncs, for a backend with no device clock to read. That one does wait,
    /// and an inner window's syncs are charged to the outer.
    ///
    /// # Errors
    ///
    /// The device refused to open or close the window, or work inside it
    /// failed and took the measurement with it. `func` has run by then; its
    /// output is lost with the error, as it would be on the read that the
    /// failure surfaces on without a window.
    fn profile<O: Send + 'static>(
        device: &Self::Device,
        name: &str,
        options: ProfileOptions,
        func: impl FnOnce() -> O + Send,
    ) -> Result<(O, ProfileDuration), ExecutionError> {
        let _ = (name, options);
        profile_system_time::<Self, O>(device, func)
    }

    /// Open a [profiling window](Self::profile) at the calling stream's
    /// current position, to be closed with
    /// [`profile_end`](Self::profile_end) from the same stream.
    ///
    /// For a caller that cannot bracket the work in a closure: a backend that
    /// forwards operations to be executed on another thread opens and closes
    /// the window from that thread, in order with the operations.
    ///
    /// `None` from a backend that opens no windows and measures only with
    /// [`profile`](Self::profile) — the default — so the caller can bracket
    /// with [`profile_system_time`] instead.
    fn profile_start(_device: &Self::Device) -> Result<Option<ProfileToken>, ExecutionError> {
        Ok(None)
    }

    /// Close the window `token` at the calling stream's current position.
    ///
    /// When `options` flush, the work the backend still holds queued for the
    /// stream executes first, so it falls inside the window. A backend that
    /// forwards the close passes `options` along, so a queue further down
    /// the chain — a remote server's fusion, say — is flushed too.
    ///
    /// Errors on a backend whose [`profile_start`](Self::profile_start) hands
    /// out no token.
    fn profile_end(
        _device: &Self::Device,
        _token: ProfileToken,
        _options: ProfileOptions,
    ) -> Result<ProfileDuration, ExecutionError> {
        Err(profile_unsupported())
    }

    /// Drop the window `token` opened without measuring it, for a caller that
    /// will never reach [`profile_end`](Self::profile_end).
    ///
    /// **An open window is not free**, and the cost is not paid once: a
    /// backend holds a start event, keeps timestamp writes on, or retains
    /// command buffers for as long as one is open, and on wgpu every later
    /// pass keeps rewriting the live window's end slot. So a window whose
    /// caller unwound between the two calls is abandoned rather than left,
    /// which is what [`profile_with_tokens`] does on the panic path.
    ///
    /// Cannot fail and answers nothing: it is called while a panic is already
    /// unwinding, where there is nobody left to tell. The default closes the
    /// window and discards the measurement, which every backend can already
    /// do; one that can drop a window without recording an end does that
    /// instead.
    fn profile_abandon(device: &Self::Device, token: ProfileToken) {
        let _ = Self::profile_end(device, token, ProfileOptions::default());
    }

    /// Prepare `device` for an upcoming graph capture: route allocations into a
    /// stable pool so every buffer allocated before graph_stop_capture can
    /// be pinned. Call before the warmup run. No-op by default.
    ///
    /// See [`burn_graph`](crate) — the closure-based `capture` helper drives
    /// this whole sequence.
    fn graph_prepare(_device: &Self::Device) -> Result<(), ExecutionError> {
        Ok(())
    }

    /// Begin recording launches on `device` into a graph (see
    /// [`graph_stop_capture`](Backend::graph_stop_capture)). Errors on backends
    /// without hardware graph support, so callers fall back to re-running.
    fn graph_start_capture(_device: &Self::Device) -> Result<(), ExecutionError> {
        Err(graph_unsupported())
    }

    /// Stop recording and return the captured [graph](BackendTypes::GraphPrimitive),
    /// ready to [`graph_replay`](Backend::graph_replay).
    fn graph_stop_capture(_device: &Self::Device) -> Result<BackendGraph<Self>, ExecutionError> {
        Err(graph_unsupported())
    }

    /// Replay a captured [graph](BackendTypes::GraphPrimitive) — one dispatch
    /// re-running the recorded launches against their original buffers.
    ///
    /// # Safety
    ///
    /// The replay dispatches raw device work against the exact buffers recorded
    /// at capture time, with nothing tracking whether those buffers are still
    /// valid. The caller must guarantee, for every tensor the captured closure
    /// read or wrote:
    ///
    /// - its buffer is still alive — no tensor referenced by the graph has been
    ///   freed (and its memory possibly reallocated) since capture;
    /// - it is not concurrently read or written by work on another stream or
    ///   thread while the replay executes;
    /// - input refreshes and output reads are issued on the stream the graph
    ///   was captured on, so they order correctly against the replay.
    unsafe fn graph_replay(
        _device: &Self::Device,
        _graph: &BackendGraph<Self>,
    ) -> Result<(), ExecutionError> {
        Err(graph_unsupported())
    }

    /// Flush any pending operation of the backend.
    fn flush(_device: &Self::Device);

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

    /// Whether the type is fully supported by the specified device for general operations.
    ///
    /// A type is considered supported if it can be used for the full suite of tensor
    /// operations, including storage, conversion, and basic arithmetic.
    ///
    /// Returning `false` does not necessarily mean the device cannot handle the type at all.
    /// For instance, a device might support a type only for specialized hardware
    /// acceleration (e.g., matrix multiplication) but lack general arithmetic support. Such
    /// types should return `false` here as they are not globally supported.
    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        Self::dtype_usage(device, dtype).is_superset(DTypeUsage::general())
    }

    /// Returns the [DTypeUsageSet] for the given [DType] on the specified device.
    fn dtype_usage(device: &Self::Device, dtype: DType) -> DTypeUsageSet;

    /// Returns the number of devices available on this backend.
    /// `device` is a reference device used to determine the underlying backend that should be queried.
    /// A CUDA device will return all devices available to CUDA, a Vulkan device will return all
    /// devices available to Vulkan, etc.
    fn device_count(type_id: u16) -> usize;
}

/// Trait that allows a backend to support autodiff.
pub trait AutodiffBackend: Backend {
    /// The inner backend type.
    type InnerBackend: Backend<Device = Self::Device>;

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

    /// Mark the tensor as distributed across multiple devices.
    /// The gradients will be aggregated during the backward pass.
    ///
    /// This function does nothing when distributed training is not available.
    fn set_distributed_params(
        tensor: FloatTensor<Self>,
        _param_id: DistributedParamId,
    ) -> FloatTensor<Self> {
        tensor
    }

    /// Returns the distributed parameters if the tensor was marked as distributed.
    fn distributed_params(_tensor: &FloatTensor<Self>) -> Option<DistributedParams> {
        None
    }

    /// Returns true if the tensor was marked as distributed.
    fn is_distributed(_tensor: &FloatTensor<Self>) -> bool {
        false
    }
}

/// Describes how a data type can be used on a given device.
///
/// A data type may be supported for different classes of operations. Not all
/// data types that appear in hardware or kernel implementations are suitable
/// for general-purpose tensor operations.
#[derive(Debug, EnumSetType)]
pub enum DTypeUsage {
    /// The type can be stored in device memory and converted to and from
    /// other supported data types.
    Storage,
    /// The type supports general-purpose arithmetic and common tensor
    /// operations (e.g. elementwise ops, reductions, etc.).
    Arithmetic,
    /// The type is supported by hardware-accelerated execution paths.
    ///
    /// This typically indicates support for accelerator-backed compute units (e.g., tensor
    /// cores executing MMA instructions) for high-performance operations such as matrix
    /// multiplication and operations that lower to it.
    ///
    /// # Notes
    /// - A type can be both [`Arithmetic`](DTypeUsage::Arithmetic) and
    ///   [`Accelerated`](DTypeUsage::Accelerated) if it supports general-purpose operations
    ///   *and* accelerated paths.
    /// - If a type is marked as `Accelerated` but not `Arithmetic`, it is not
    ///   suitable for general-purpose tensor operations and may only be used
    ///   in specific accelerated operations.
    ///
    /// `Accelerated` is a **flag**, not a detailed descriptor. It does not enumerate which
    /// operations are accelerated or which accelerator features are available.
    Accelerated,
}

/// A set of [DTypeUsage] representing the total capabilities of a data type on a device.
pub type DTypeUsageSet = EnumSet<DTypeUsage>;

impl DTypeUsage {
    /// Returns the usage set required for general-purpose tensor support.
    pub fn general() -> DTypeUsageSet {
        DTypeUsage::Storage | DTypeUsage::Arithmetic
    }
}
