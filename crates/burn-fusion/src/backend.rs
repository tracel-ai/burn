use crate::{
    client::FusionClient, stream::Context, FusionClientLocator, FusionTensor, PrecisionBridge,
    QFusionTensor,
};
use burn_common::stream::StreamId;
use burn_tensor::{
    backend::{Backend, DeviceOps, SyncType},
    ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
    repr::{OperationDescription, ReprBackend, TensorHandle},
    Device, Element,
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

    type FloatTensorPrimitive = FusionTensor<B::FusionRuntime>;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive = FusionTensor<B::FusionRuntime>;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive = FusionTensor<B::FusionRuntime>;

    type QuantizedTensorPrimitive = QFusionTensor<B::FusionRuntime>;

    type QuantizedEncoding = B::QuantizedEncoding;

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
    type FusionHandle: Clone + Send + Sync;
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
    fn cast_float(tensor: FloatTensor<Self>, dtype: burn_tensor::DType) -> Self::Handle;

    /// Pointer to the full precision fusion backend.
    type FullPrecisionBackend: FusionBackend<FusionRuntime = Self::FusionRuntime>;
}

// Fusion implements `ReprBackend` to enable router backend usage.
impl<B: FusionBackend> ReprBackend for Fusion<B> {
    type Handle = B::Handle; // aka JitFusionHandle

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        let primitive = B::float_tensor(handle.clone());
        let device = B::float_device(&primitive);

        let shape = handle.shape;
        let client = get_client::<B>(&device);
        client.register_tensor(
            handle.handle,
            shape.dims,
            StreamId::current(),
            B::FloatElem::dtype(),
        )
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        let primitive = B::int_tensor(handle.clone());
        let device = B::int_device(&primitive);

        let shape = handle.shape;
        let client = get_client::<B>(&device);
        client.register_tensor(
            handle.handle,
            shape.dims,
            StreamId::current(),
            B::IntElem::dtype(),
        )
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        let primitive = B::bool_tensor(handle.clone());
        let device = B::bool_device(&primitive);

        let shape = handle.shape;
        let client = get_client::<B>(&device);
        client.register_tensor(
            handle.handle,
            shape.dims,
            StreamId::current(),
            burn_tensor::DType::Bool,
        )
    }

    fn quantized_tensor(
        handles: Vec<TensorHandle<Self::Handle>>,
        scheme: burn_tensor::quantization::QuantizationScheme,
    ) -> QuantizedTensor<Self> {
        todo!() // not as simple
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        todo!()
        // gotta go from FusionTensor <-> JitTensor (aka B::FloatTensorPrimitive)
        // maybe into_description -> handle
        // need to access handle container for that!
        // FusionServer has HandleContainer<R::FusionHandle> where R: FusionRuntime
        // The FusionServer is in MutexFusionClient (behind Arc<Mutex<..>>)
        // MutexFusionClient implements FusionClient
        // B::float_tensor_handle(tensor) // doesn't work because `tensor` is not B::FloatTensorPrimitive but a FusionTensor instead
        // .. and from what I understand B::FloatTensorPrimitive is only accessible in an operation description (for kernel execution)
        // so how tf can this be done?
        // let client = tensor.client.clone();

        // client.get_tensor_handle(&tensor.into_description())
        // TODO: client.get_tensor_handle() will fail when trying to get an uninitialized output tensor handle
        // so we have to find a way to get the tensor handle without
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        // let client = tensor.client.clone();
        // client.get_tensor_handle(&tensor.into_description())
        todo!()
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        // let client = tensor.client.clone();
        // client.get_tensor_handle(&tensor.into_description())
        todo!()
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Vec<Self::Handle> {
        todo!() // not as simple
    }
}
