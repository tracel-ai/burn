use crate::{
    client::FusionClient, stream::Context, FusionClientLocator, FusionTensor, PrecisionBridge,
};
use burn_tensor::{
    backend::{Backend, DeviceOps, SyncType},
    ops::FloatTensor,
    repr::{OperationDescription, ReprBackend},
    Device,
};
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;

pub(crate) static CLIENTS: FusionClientLocator = FusionClientLocator::new();

pub(crate) fn get_client<B: FusionBackend>(device: &Device<B>) -> Client<B::FusionRuntime> {
    CLIENTS.client::<B::FusionRuntime>(device)
}

/// Enable dynamic operation fusion on a backend that implements [fusion backend](crate::FusionBackend).
#[derive(Clone, Debug, Default)]
pub struct Fusion<B: FusionBackend> {
    _backend: PhantomData<B>,
}

impl<B: FusionBackend> Backend for Fusion<B> {
    type Device = B::Device;

    type FullPrecisionBridge = PrecisionBridge<B::FullPrecisionBackend>;

    type FloatTensorPrimitive<const D: usize> = FusionTensor<B::FusionRuntime>;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive<const D: usize> = FusionTensor<B::FusionRuntime>;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive<const D: usize> = FusionTensor<B::FusionRuntime>;

    fn name() -> String {
        format!("fusion<{}>", B::name())
    }

    fn seed(seed: u64) {
        B::seed(seed);
    }

    fn sync(device: &Self::Device, sync_type: SyncType) {
        let client = CLIENTS.client::<B::FusionRuntime>(&device.clone());
        client.drain();
        B::sync(device, sync_type);
    }

    fn ad_enabled() -> bool {
        false
    }
}

/// The status of a [builder](OptimizationBuilder).
#[derive(Clone, Debug, Copy)]
pub enum OptimizationStatus {
    /// No more operations can be fused.
    Closed,
    /// More operations can be fused.
    Open,
}

/// The properties of a [builder](OptimizationProperties).
#[derive(Debug, Clone, Copy, Default)]
pub struct OptimizationProperties {
    /// The score of the optimization, higher is better.
    pub score: u64,
    /// If the operation is ready to be executed.
    pub ready: bool,
}

/// The fusion operation abstraction allows implementations to fuse many
/// [tensor operations](OperationDescription) into one, improving the performance of the backend.
///
///
/// # Notes
///
/// The implementations are free to execute the registered operations the way they want to improve
/// the speed and efficiency of the computational graph. It doesn't mean that all registered
/// operations should be fused, but that another way of executing them is more efficient.
///
/// Also, it is important to return (OptimizationStatus::Closed) when no more registered operation can
/// improve the performance.
pub trait OptimizationBuilder<O>: Send {
    /// Register a new [tensor operation](OperationDescription).
    fn register(&mut self, operation: &OperationDescription);
    /// Finish the optimization and create a fusion operation.
    fn build(&self) -> O;
    /// Reset the state.
    fn reset(&mut self);
    /// Return the builder [status](OptimizationStatus).
    fn status(&self) -> OptimizationStatus;
    /// Return the builder [properties](OptimizationProperties).
    fn properties(&self) -> OptimizationProperties;
    /// The number of operation fused.
    fn len(&self) -> usize;
    /// If no operations are fused.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The operation created from the [builder](OptimizationBuilder).
pub trait Optimization<R: FusionRuntime>: Send {
    /// Execute the operation.
    fn execute(&mut self, context: &mut Context<'_, R::FusionHandle>);
    /// The number of registered operations in this optimization.
    fn len(&self) -> usize;
    /// If the current optimization is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the state that can be serialized.
    fn to_state(&self) -> R::OptimizationState;
    /// Create the optimization from the state.
    fn from_state(device: &R::FusionDevice, state: R::OptimizationState) -> Self;
}

/// Type alias for `<R as FusionRuntime>::FusionDevice`.
pub type FusionDevice<R> = <R as FusionRuntime>::FusionDevice;
/// Type alias for `<R as FusionRuntime>::FusionHandle`.
pub type FusionHandle<R> = <R as FusionRuntime>::FusionHandle;
/// Type alias for `<R as FusionRuntime>::FusionClient`.
pub type Client<R> = <R as FusionRuntime>::FusionClient;

/// Trait that defines a runtime that will benefits from fused operations.
pub trait FusionRuntime: Send + Sync + Sized + core::fmt::Debug {
    /// The state that can be serialized for an optimization.
    type OptimizationState: Serialize + DeserializeOwned;
    /// Optimization type for the backend.
    type Optimization: Optimization<Self>;
    /// Handle used to store tensor dynamically.
    type FusionHandle: Clone + Send;
    /// Device used by the runtime.
    type FusionDevice: DeviceOps;
    /// The client to interact with the runtime.
    type FusionClient: FusionClient<Self>;

    /// The list of optimizations that will be used to optimize the computational graph.
    fn optimizations(
        device: Self::FusionDevice,
    ) -> Vec<Box<dyn OptimizationBuilder<Self::Optimization>>>;
}

/// Trait that allows an existing [backend](Backend) to specify graph optimizations using
/// [operation builder](crate::OptimizationBuilder).
pub trait FusionBackend:
    ReprBackend<Handle = FusionHandle<Self::FusionRuntime>, Device = FusionDevice<Self::FusionRuntime>>
{
    /// The runtime used for this backend.
    type FusionRuntime: FusionRuntime;

    /// Cast a float tensor and returns the resulting handle.
    fn cast_float<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dtype: burn_tensor::DType,
    ) -> Self::Handle;

    /// Pointer to the full precision fusion backend.
    type FullPrecisionBackend: FusionBackend<FusionRuntime = Self::FusionRuntime>;
}
