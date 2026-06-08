use crate::{
    FusionTensor,
    client::GlobalFusionClient,
    stream::{Context, OrderedExecution},
};
use burn_backend::{
    Backend, BackendTypes, DType, DeviceOps, ExecutionError,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_ir::{BackendIr, OperationIr, TensorHandle};
use serde::{Serialize, de::DeserializeOwned};
use std::marker::PhantomData;

/// Get the client for the given device.
pub fn get_client<B: FusionBackend>(device: &Device<B>) -> Client<B::FusionRuntime> {
    GlobalFusionClient::load(device)
}

/// Enable dynamic operation fusion on a backend that implements [fusion backend](crate::FusionBackend).
#[derive(Clone, Debug, Default)]
pub struct Fusion<B: FusionBackend> {
    _backend: PhantomData<B>,
}

impl<B: FusionBackend> BackendTypes for Fusion<B> {
    type Device = B::Device;

    type FloatTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type IntTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type BoolTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type QuantizedTensorPrimitive = FusionTensor<B::FusionRuntime>;
}

impl<B: FusionBackend> Backend for Fusion<B> {
    fn name(device: &Self::Device) -> String {
        format!("fusion<{}>", B::name(device))
    }

    fn seed(device: &B::Device, seed: u64) {
        let client = GlobalFusionClient::<B::FusionRuntime>::load(device);
        let device = device.clone();
        client.sync(move || B::seed(&device, seed));
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = GlobalFusionClient::<B::FusionRuntime>::load(device);
        let device = device.clone();
        client.sync(move || B::sync(&device))
    }

    fn ad_enabled(_device: &Self::Device) -> bool {
        false
    }

    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        B::memory_persistent_allocations(device, input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        B::memory_cleanup(device)
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut burn_backend::TensorData>,
    {
        B::staging(data, device);
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        B::supports_dtype(device, dtype)
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> burn_backend::DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }

    fn device_count(type_id: u16) -> usize {
        B::device_count(type_id)
    }
}

/// The status of a [fuser](OperationFuser).
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum FuserStatus {
    /// No more operations can be fused.
    Closed,
    /// More operations can be fused.
    Open,
}

/// The properties of a [fuser](OperationFuser).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FuserProperties {
    /// The score of the optimization, higher is better.
    pub score: u64,
    /// If the operation is ready to be executed.
    pub ready: bool,
}

/// The fusion operation abstraction allows implementations to fuse many
/// [tensor operations](OperationIr) into one, improving the performance of the backend.
///
///
/// # Notes
///
/// The implementations are free to execute the registered operations the way they want to improve
/// the speed and efficiency of the computational graph. It doesn't mean that all registered
/// operations should be fused, but that another way of executing them is more efficient.
///
/// Also, it is important to return (FuserStatus::Closed) when no more registered operation can
/// improve the performance.
pub trait OperationFuser<O>: Send {
    /// Register a new [tensor operation](OperationIr).
    fn fuse(&mut self, operation: &OperationIr);
    /// Finish the optimization and create a fusion operation.
    fn finish(&mut self) -> O;
    /// Reset the state.
    fn reset(&mut self);
    /// Return the builder [status](FuserStatus).
    fn status(&self) -> FuserStatus;
    /// Return the builder [properties](FuserProperties).
    fn properties(&self) -> FuserProperties;
    /// The number of operation fused.
    fn len(&self) -> usize;
    /// If no operations are fused.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Clone the optimization builder.
    fn clone_dyn(&self) -> Box<dyn OperationFuser<O>>;
}

/// The number of operations contained in the data structure.
pub trait NumOperations: core::fmt::Debug {
    /// The number of registered operations.
    fn len(&self) -> usize;
    /// If the current optimization is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The name of the optimization.
    fn name(&self) -> &'static str;
}

/// The optimization created from a [fuser](OperationFuser).
pub trait Optimization<R: FusionRuntime>: Send + NumOperations {
    /// Execute the optimization.
    fn execute(&mut self, context: &mut Context<R::FusionHandle>, execution: &OrderedExecution<R>);

    /// Returns the state that can be serialized.
    fn to_state(&self) -> R::OptimizationState;
    /// Create the optimization from the state.
    fn from_state(device: &R::FusionDevice, state: R::OptimizationState) -> Self;
}

/// Type alias for `<R as FusionRuntime>::FusionDevice`.
pub type FusionDevice<R> = <R as FusionRuntime>::FusionDevice;
/// Type alias for `<R as FusionRuntime>::FusionHandle`.
pub type FusionHandle<R> = <R as FusionRuntime>::FusionHandle;
/// Client alias.
pub type Client<R> = GlobalFusionClient<R>;

/// Trait that defines a runtime that will benefits from fused operations.
pub trait FusionRuntime: Send + Sync + Sized + core::fmt::Debug + 'static {
    /// The state that can be serialized for an optimization.
    type OptimizationState: Serialize + DeserializeOwned;
    /// Optimization type for the backend.
    type Optimization: Optimization<Self>;
    /// Handle used to store tensor dynamically.
    type FusionHandle: Clone + Send;
    /// Device used by the runtime.
    type FusionDevice: DeviceOps;

    /// The list of fusers that will be used to optimize the computational graph.
    fn fusers(device: Self::FusionDevice) -> Vec<Box<dyn OperationFuser<Self::Optimization>>>;
}

/// Bound that adds a [`DistributedBackend`](burn_backend::distributed::DistributedBackend)
/// requirement when the `distributed` feature is enabled, and is a no-op otherwise.
///
/// Abstracting the conditional supertrait here lets [`FusionBackend`] keep a single declaration
/// regardless of the feature flag, while still guaranteeing that the remote/router interpreter
/// path ([`BackendIr::register_distributed`]) can drive collective operations through fusion. The
/// only in-tree fusion backend (`CubeBackend`) satisfies it unconditionally.
#[cfg(feature = "distributed")]
pub trait MaybeDistributedBackend: burn_backend::distributed::DistributedBackend {}
#[cfg(feature = "distributed")]
impl<B: burn_backend::distributed::DistributedBackend> MaybeDistributedBackend for B {}

/// Bound that adds a `DistributedBackend` requirement when the `distributed` feature is enabled,
/// and is a no-op otherwise. See the `distributed`-enabled definition for details.
#[cfg(not(feature = "distributed"))]
pub trait MaybeDistributedBackend {}
#[cfg(not(feature = "distributed"))]
impl<B> MaybeDistributedBackend for B {}

/// Trait that allows an existing [backend](Backend) to specify graph optimizations using
/// [operation fuser](crate::OperationFuser).
pub trait FusionBackend:
    BackendIr<
        Handle = FusionHandle<Self::FusionRuntime>,
        Device = FusionDevice<Self::FusionRuntime>,
    > + MaybeDistributedBackend
{
    /// The runtime used for this backend.
    type FusionRuntime: FusionRuntime;

    /// Cast a float tensor and returns the resulting handle.
    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle;

    /// Pointer to the full precision fusion backend.
    type FullPrecisionBackend: FusionBackend<FusionRuntime = Self::FusionRuntime>;
}

// Fusion implements `BackendIr` to enable router backend usage.
impl<B: FusionBackend> BackendIr for Fusion<B> {
    type Handle = FusionTensor<B::FusionRuntime>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        handle.handle
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        handle.handle
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        handle.handle
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        handle.handle
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor
    }

    // The `MaybeDistributedBackend` supertrait on `FusionBackend` guarantees `Self:
    // DistributedBackend` under the `distributed` feature, so the remote/router interpreter can
    // drive collectives through fusion just like the cube backend. `all_reduce` registers the op
    // into the fusion stream; we resolve it eagerly here because the interpreter hands back a
    // concrete handle.
    #[cfg(feature = "distributed")]
    fn register_distributed(
        op: &burn_ir::DistributedOperationIr,
        device: &Self::Device,
        handles: &mut burn_ir::HandleContainer<Self::Handle>,
    ) {
        use burn_backend::distributed::DistributedBackend;

        match op {
            burn_ir::DistributedOperationIr::AllReduce(desc) => {
                let tensor = handles.get_float_tensor::<Self>(&desc.tensor);
                let device_ids = desc.device_ids.iter().map(|id| (*id).into()).collect();
                let output = Self::all_reduce(tensor, desc.op, device_ids);
                // Safety: immediately registered, not accessed before the collective resolves.
                let output = unsafe { output.assume_resolved() };
                handles.register_float_tensor::<Self>(&desc.out.id, output);
            }
            burn_ir::DistributedOperationIr::SyncCollective => Self::sync_collective(device),
        }
    }
}
