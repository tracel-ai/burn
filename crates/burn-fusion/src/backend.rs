use crate::{
    client::FusionClient, stream::Context, FusionClientLocator, FusionTensor, PrecisionBridge,
};
use burn_tensor::{
    backend::Backend,
    repr::{OperationDescription, ReprBackend},
    Device,
};
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;

pub(crate) static CLIENTS: FusionClientLocator = FusionClientLocator::new();

pub(crate) fn get_client<B: FusionBackend>(device: &B::Device) -> B::FusionClient {
    CLIENTS.client(device)
}

/// Enable dynamic operation fusion on a backend that implements [fusion backend](crate::FusionBackend).
#[derive(Clone, Debug, Default)]
pub struct Fusion<B> {
    _backend: PhantomData<B>,
}

impl<B: FusionBackend> Backend for Fusion<B> {
    type Device = B::Device;

    type FullPrecisionBridge = PrecisionBridge;

    type FloatTensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    fn name() -> String {
        format!("fusion<{}>", B::name())
    }

    fn seed(seed: u64) {
        B::seed(seed);
    }

    fn sync(device: &Self::Device) {
        let client = CLIENTS.client::<B::FusionClient>(&device.clone());
        client.drain();
        B::sync(device)
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
pub trait Optimization<B: FusionBackend>: Send {
    /// Execute the operation.
    fn execute(&mut self, context: &mut Context<'_, B>);
    /// The number of registered operations in this optimization.
    fn len(&self) -> usize;
    /// If the current optimization is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the state that can be serialized.
    fn to_state(&self) -> B::OptimizationState;
    /// Create the optimization from the state.
    fn from_state(device: &B::Device, state: B::OptimizationState) -> Self;
}

/// Trait that allows an existing [backend](Backend) to specify graph optimizations using
/// [operation builder](crate::OptimizationBuilder).
pub trait FusionBackend: Backend + ReprBackend {
    /// The state that can be serialized for an optimization.
    type OptimizationState: Serialize + DeserializeOwned;
    /// Optimization type for the backend.
    type Optimization: Optimization<Self>;
    /// What kind of client should be used.
    type FusionClient: FusionClient<FusionBackend = Self>;

    /// The list of optimizations that will be used to optimize the computational graph.
    fn optimizations(device: Device<Self>)
        -> Vec<Box<dyn OptimizationBuilder<Self::Optimization>>>;
}
