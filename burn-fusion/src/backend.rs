use crate::{
    client::FusionClient,
    graph::{Context, OptimizationFactory, TensorOpsDescription},
    FusionClientLocator, FusionTensor,
};
use burn_tensor::{backend::Backend, Device, Shape};
use std::marker::PhantomData;

pub(crate) static CLIENTS: FusionClientLocator = FusionClientLocator::new();

pub(crate) fn get_client<B: FusionBackend>(device: &B::FusionDevice) -> B::FusionClient {
    CLIENTS.client(device)
}

/// Enable dynamic operation fusion on a backend that implements [fusion backend](crate::FusionBackend).
#[derive(Clone, Debug, Default)]
pub struct Fusion<B> {
    _backend: PhantomData<B>,
}

impl<B: FusionBackend> Backend for Fusion<B> {
    type Device = B::Device;

    // TODO: Find a better way to handle full precision.
    type FullPrecisionBackend = Self;
    type FullPrecisionElem = B::FloatElem;

    type TensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

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
        let client = CLIENTS.client::<B::FusionClient>(&device.clone().into());
        client.drain_graph();
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
/// [tensor operations](TensorOpsDescription) into one, improving the performance of the backend.
///
///
/// # Notes
///
/// The implementations are free to execute the registered operations the way they want to improve
/// the speed and efficiency of the computational graph. It doesn't mean that all registered
/// operations should be fused, but that another way of executing them is more efficient.
///
/// Also, it is important to return (FusionStatus::Closed) when no more registered operation can
/// improve the performance.
pub trait OptimizationBuilder<B: FusionBackend>: Send {
    /// Register a new [tensor operation](TensorOpsDescription).
    fn register(&mut self, ops: &TensorOpsDescription);
    /// Finish the optimization and create a fusion operation.
    fn build(&self) -> Box<dyn Optimization<B>>;
    /// Reset the state.
    fn reset(&mut self);
    /// Return the builder [status](OptimizationStatus).
    fn status(&self) -> OptimizationStatus;
    /// Return the builder [properties](OptimizationProperties).
    fn properties(&self) -> OptimizationProperties;
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
}

// We implement the OptimizationFactory for all boxed optimization to be used with the Optimization
// Cache. The factory is only used to simplify types and allows better testing. It isn't a public
// crate.
impl<B: FusionBackend> OptimizationFactory<Box<dyn Optimization<B>>>
    for Box<dyn OptimizationBuilder<B>>
{
    fn create(&self) -> Box<dyn Optimization<B>> {
        OptimizationBuilder::build(self.as_ref())
    }
}

/// The device id.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The type id identifies the type of the device.
    pub type_id: u16,
    /// The index id identifies the device number.
    pub index_id: u32,
}

/// The handle device trait allows to get an id for a backend device.
pub trait FusionDevice: Clone + Send + Sync + PartialEq {
    /// Return the [device id](DeviceId).
    fn id(&self) -> DeviceId;
}

/// Trait that allows an existing [backend](Backend) to specify graph optimizations using
/// [operation builder](crate::OptimizationBuilder).
pub trait FusionBackend: Backend {
    /// The device type that can return an ID.
    ///
    /// It can be the same as (Backend::Device), but must implement (FusionDevice).
    type FusionDevice: FusionDevice + From<Self::Device> + Into<Self::Device> + core::fmt::Debug;
    /// The type that can be used to point to a tensor of any kind.
    type Handle: Sync + Send + Clone;
    /// What kind of client should be used.
    type FusionClient: FusionClient<FusionBackend = Self>;

    /// The list of optimizations that will be used to optimize the computational graph.
    fn optimizations(device: &Device<Self>) -> Vec<Box<dyn OptimizationBuilder<Self>>>;

    /// Convert a [handle](FusionBackend::Handle) to a [float tensor](Backend::TensorPrimitive).
    fn float_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::TensorPrimitive<D>;
    /// Convert a [handle](FusionBackend::Handle) to an [int tensor](Backend::IntTensorPrimitive).
    fn int_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::IntTensorPrimitive<D>;
    /// Convert a [handle](FusionBackend::Handle) to a [bool tensor](Backend::BoolTensorPrimitive).
    fn bool_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> Self::BoolTensorPrimitive<D>;

    /// Convert a [float tensor](Backend::TensorPrimitive) to a [handle](FusionBackend::Handle).
    fn float_tensor_handle<const D: usize>(tensor: Self::TensorPrimitive<D>) -> Self::Handle;
    /// Convert an [int tensor](Backend::IntTensorPrimitive) to a [handle](FusionBackend::Handle).
    fn int_tensor_handle<const D: usize>(tensor: Self::IntTensorPrimitive<D>) -> Self::Handle;
    /// Convert a [bool tensor](Backend::BoolTensorPrimitive) to a [handle](FusionBackend::Handle).
    fn bool_tensor_handle<const D: usize>(tensor: Self::BoolTensorPrimitive<D>) -> Self::Handle;
}
