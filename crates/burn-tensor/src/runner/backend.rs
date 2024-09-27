use core::{future::Future, marker::PhantomData};
use std::collections::HashMap;

use burn_common::{stream::StreamId, stub::Mutex};

use crate::{
    backend::{Backend, BackendBridge, DeviceId, DeviceOps},
    ops::FloatTensor,
    quantization::QTensorPrimitive,
    repr::{HandleContainer, OperationDescription, ReprBackend, TensorDescription},
    DType, Device, TensorData,
};

use super::RouterTensor;

// Inspiration fusion: type MyBackend = Fusion<JitBackend<WgpuRuntime, F, I>>;
// Example usage: type MultiBackend = Runner<[Insert RunnerBackend here]>?
// - What should the RunnerBackend trait be implemented on? Each backend that can be used in a multi-backend setting?
// - And the RunnerRuntime?

// ~~Runner~~ BackendRouter types examples: HttpRouter, InMemoryRouter, GrpcRouter

pub struct BackendRouter<R: RunnerChannel> {
    r: PhantomData<R>,
}

// pub type Client<B: BackendChannel> = <B::Runtime as RunnerRuntime>::Client;

// Allow switching the backend dynamically using a device.
// pub trait BackendChannel: Clone + Send + Sync + 'static + Sized {
//     type Runtime: RunnerRuntime;
//     /// Bridge to transfer tensors between backends.
//     // type Bridge: BackendBridge<BackendRouter<Self>> + 'static;

//     /// Get the runner client for the given device.
//     fn client(device: &Device<Runner<Self>>) -> Client<Self>;
// }

// pub trait RunnerRuntime {
//     type Client: RunnerClient;
//     type Device: DeviceOps;
// }

impl<R: RunnerChannel> core::fmt::Debug for BackendRouter<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("router"))
    }
}

impl<R: RunnerChannel> Clone for BackendRouter<R> {
    fn clone(&self) -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RunnerChannel> Default for BackendRouter<R> {
    fn default() -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RunnerChannel> QTensorPrimitive for RouterTensor<R> {
    fn scheme(&self) -> &crate::quantization::QuantizationScheme {
        todo!()
    }

    fn strategy(&self) -> crate::quantization::QuantizationStrategy {
        todo!()
    }
}

/// Define how to interact with the runner server.
pub trait RunnerClient: Clone + Send + Sync + Sized {
    /// Register a new tensor operation to be executed by the (runner) server.
    fn register(&self, op: OperationDescription, stream: StreamId);
    /// Read the values contained by a tensor.
    fn read_tensor(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send;
    fn write_tensor(&self, data: TensorData, stream: StreamId) -> TensorDescription;
    fn empty_tensor(&self, shape: Vec<usize>, dtype: DType, stream: StreamId) -> TensorDescription;
}

/// Handle precision conversion.
#[derive(Debug)]
pub struct PrecisionBridge {}

impl<R: RunnerChannel> BackendBridge<BackendRouter<R>> for PrecisionBridge {
    type Target = BackendRouter<R>;

    fn into_target(
        tensor: FloatTensor<BackendRouter<R>>,
        _device: Option<<BackendRouter<R> as Backend>::Device>,
    ) -> FloatTensor<Self::Target> {
        todo!()
        // TODO: smilar to fusion `cast` in burn-fusion/src/bridge.rs
    }

    fn from_target(
        tensor: FloatTensor<Self::Target>,
        _device: Option<<BackendRouter<R> as Backend>::Device>,
    ) -> FloatTensor<BackendRouter<R>> {
        todo!()
    }
}

impl<C: RunnerChannel> Backend for BackendRouter<C> {
    type Device = C::Device;

    type FullPrecisionBridge = PrecisionBridge;

    type FloatTensorPrimitive = RouterTensor<C>;

    // TODO: how to set elem types?
    type FloatElem = f32;

    type IntTensorPrimitive = RouterTensor<C>;

    type IntElem = i32;

    type BoolTensorPrimitive = RouterTensor<C>;

    type QuantizedTensorPrimitive = RouterTensor<C>;

    type QuantizedEncoding = u32;

    fn name() -> String {
        todo!()
    }

    fn seed(seed: u64) {
        todo!()
    }
}

pub struct RunnerContext<B: ReprBackend> {
    handles: HandleContainer<B::Handle>,
    device: B::Device,
}

// Runners are used by the channels (e.g., DirectChannel, HttpChannel, etc.)
// Responsible for executing the tensor operations.
#[derive(Clone)]
pub struct Runner<B: ReprBackend> {
    context: Mutex<RunnerContext<B>>,
}

impl<B: ReprBackend> Runner<B> {
    fn new(device: B::Device) -> Self {
        Self {
            context: RunnerContext {
                handles: HandleContainer::new(),
                device,
            },
        }
    }
}

impl<B: ReprBackend> RunnerClient for Runner<B> {
    /// Execute a tensor operation.
    fn register(&self, op: crate::repr::OperationDescription, stream: StreamId) {
        match op {
            OperationDescription::BaseFloat(_) => todo!(),
            OperationDescription::BaseInt(_) => todo!(),
            OperationDescription::BaseBool(_) => todo!(),
            OperationDescription::NumericFloat(_, _) => todo!(),
            OperationDescription::NumericInt(_, _) => todo!(),
            OperationDescription::Bool(_) => todo!(),
            OperationDescription::Int(_) => todo!(),
            OperationDescription::Float(_dtype, op) => match op {
                crate::repr::FloatOperationDescription::Exp(_) => todo!(),
                crate::repr::FloatOperationDescription::Log(_) => todo!(),
                crate::repr::FloatOperationDescription::Log1p(_) => todo!(),
                crate::repr::FloatOperationDescription::Erf(_) => todo!(),
                crate::repr::FloatOperationDescription::PowfScalar(_) => todo!(),
                crate::repr::FloatOperationDescription::Sqrt(_) => todo!(),
                crate::repr::FloatOperationDescription::Cos(_) => todo!(),
                crate::repr::FloatOperationDescription::Sin(_) => todo!(),
                crate::repr::FloatOperationDescription::Tanh(_) => todo!(),
                crate::repr::FloatOperationDescription::IntoInt(_) => todo!(),
                crate::repr::FloatOperationDescription::Matmul(_) => todo!(),
                crate::repr::FloatOperationDescription::Random(_desc) => {
                    todo!() // B::float_random(shape, distribution, device)
                }
                crate::repr::FloatOperationDescription::Recip(_) => todo!(),
                crate::repr::FloatOperationDescription::Quantize(_) => todo!(),
                crate::repr::FloatOperationDescription::Dequantize(_) => todo!(),
            },
            OperationDescription::Module(_) => todo!(),
        }
    }

    async fn read_tensor(&self, tensor: TensorDescription, stream: StreamId) -> TensorData {
        todo!()
    }

    fn write_tensor(&self, data: TensorData, stream: StreamId) -> TensorDescription {
        todo!()
    }

    fn empty_tensor(&self, shape: Vec<usize>, dtype: DType, stream: StreamId) -> TensorDescription {
        todo!()
    }
}

// Defines associated types config for a setup with multiple backend runners.
pub trait RunnerChannel: Clone + Send + Sync + 'static + Sized {
    type Device: DeviceOps;
    type Bridge: MultiBackendBridge;
    type Client: RunnerClient;

    /// Initialize a new client for the given device.
    fn init_client(device: &Self::Device) -> Self::Client;

    /// Change the tensor to a different runner.
    fn change_runner(
        self,
        tensor: <Self::Bridge as MultiBackendBridge>::TensorType,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType {
        Self::Bridge::move_from(&self, tensor)
    }
}

// TODO: macro this shit for different number of backends (i.e., up to 6)
#[derive(Clone, Debug)]
pub enum MultiDevice2<B1: Backend, B2: Backend> {
    Device1(B1::Device),
    Device2(B2::Device),
}
impl<B1: Backend, B2: Backend> PartialEq for MultiDevice2<B1, B2> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Device1(lhs), Self::Device1(rhs)) => lhs == rhs,
            (Self::Device2(lhs), Self::Device2(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}
impl<B1: Backend, B2: Backend> Eq for MultiDevice2<B1, B2> {}

impl<B1: Backend, B2: Backend> Default for MultiDevice2<B1, B2> {
    fn default() -> Self {
        Self::Device1(B1::Device)
    }
}

impl<B1: Backend, B2: Backend> DeviceOps for MultiDevice2<B1, B2> {
    fn id(&self) -> crate::backend::DeviceId {
        match self {
            MultiDevice2::Device1(device) => device.id(),
            MultiDevice2::Device2(device) => device.id(),
        }
    }
}

pub struct DirectChannel<Backends, Bridge> {
    backends: PhantomData<Backends>,
    bridge: PhantomData<Bridge>,
}

impl<Backends, Bridge> Clone for DirectChannel<Backends, Bridge> {
    fn clone(&self) -> Self {
        Self {
            backends: self.backends.clone(),
            bridge: self.bridge.clone(),
        }
    }
}

impl<B1: ReprBackend, B2: ReprBackend, Br: MultiBackendBridge<TensorType = Handle2<B1, B2>>>
    RunnerChannel for DirectChannel<(B1, B2), Br>
{
    type Device = MultiDevice2<B1, B2>;

    type Bridge = Br;

    type Client = MultiRunnerClient2<B1, B2>;

    fn init_client(device: &Self::Device) -> Self::Client {
        match device {
            MultiDevice2::Device1(device) => {
                MultiRunnerClient2::RunnerClient1(Runner::new(device.clone()))
            }
            MultiDevice2::Device2(device) => {
                MultiRunnerClient2::RunnerClient2(Runner::new(device.clone()))
            }
        }
    }
}

pub enum Handle2<B1: Backend, B2: Backend> {
    FloatHandle1(B1::FloatTensorPrimitive),
    FloatHandle2(B2::FloatTensorPrimitive),
    IntHandle1(B1::IntTensorPrimitive),
    IntHandle2(B2::IntTensorPrimitive),
}

pub trait MultiBackendBridge: Send + Sync + 'static {
    type TensorType;
    fn move_from(&self, tensor: Self::TensorType) -> Self::TensorType;
}

#[derive(Clone)]
pub enum MultiRunnerClient2<B1: ReprBackend, B2: ReprBackend> {
    RunnerClient1(Runner<B1>),
    RunnerClient2(Runner<B2>),
}

impl<B1: ReprBackend, B2: ReprBackend> RunnerClient for MultiRunnerClient2<B1, B2> {
    fn register(&self, op: OperationDescription, stream: StreamId) {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => runner.register(op, stream),
            MultiRunnerClient2::RunnerClient2(runner) => runner.register(op, stream),
        }
    }

    async fn read_tensor(&self, tensor: TensorDescription, stream: StreamId) -> TensorData {
        todo!()
    }

    fn write_tensor(&self, data: TensorData, stream: StreamId) -> TensorDescription {
        todo!()
    }

    fn empty_tensor(&self, shape: Vec<usize>, dtype: DType, stream: StreamId) -> TensorDescription {
        todo!()
    }
}

pub struct ByteBridge<Backends> {
    backends: PhantomData<Backends>,
}

// impl< ByteBridge<()>
impl<B1: ReprBackend, B2: ReprBackend> MultiBackendBridge for ByteBridge<(B1, B2)> {
    type TensorType = Handle2<B1, B2>;

    fn move_from(&self, tensor: Self::TensorType) -> Self::TensorType {
        match tensor {
            Handle2::FloatHandle1(tensor) => {
                let data = crate::try_read_sync(B1::float_into_data(tensor)).expect(
                    "Failed to read tensor data synchronously.
                This can happen on platforms that don't support blocking futures like WASM.
                If possible, try using into_data_async instead.",
                );

                // TODO: use `B2::float_tensor(handle)`?
                Handle2::FloatHandle2(B2::float_from_data(data, device))
            }
            Handle2::FloatHandle2(tensor) => todo!(),
            Handle2::IntHandle1(tensor) => todo!(),
            Handle2::IntHandle2(tensor) => todo!(),
        }
    }
}

// type MyBackend = BackendRouter<DirectChannel<(Cuda, NdArray, Wgpu), ByteBridge<(Cuda, NdArray, Wgpu)>>>
// ByteBridge is the default bridge for moving data between backends
// For efficient data movement/transfer, you can implement your own struct

// Runner, Server, Client, Channel
// pub struct DirectChannel<C: MultiRunnerConfig> {
//     /// Runner mappings from ID -> Runner to execute tasks
//     channel: C::Channel,
// }

/// Type alias for `<R as RunnerChannel>::Client`.
pub type Client<R> = <R as RunnerChannel>::Client;
pub(crate) static CLIENTS: RunnerClientLocator = RunnerClientLocator::new();
type Key = (core::any::TypeId, DeviceId);

pub(crate) struct RunnerClientLocator {
    clients: Mutex<Option<HashMap<Key, Box<dyn core::any::Any + Send>>>>,
}

pub(crate) fn get_client<R: RunnerChannel>(device: &R::Device) -> Client<R> {
    CLIENTS.client::<R>(device)
}

impl RunnerClientLocator {
    /// Create a new client locator.
    pub const fn new() -> Self {
        Self {
            clients: Mutex::new(None),
        }
    }

    /// Get the runner client for the given device.
    ///
    /// If a client isn't already initialized, it is created.
    pub fn client<R: RunnerChannel + 'static>(&self, device: &R::Device) -> Client<R> {
        let device_id = device.id();
        let client_id = (core::any::TypeId::of::<R>(), device_id);
        let mut clients = self.clients.lock();

        if clients.is_none() {
            let client = R::init_client(device.clone());
            Self::register_inner::<R>(client_id, client, &mut clients);
        }

        match clients.deref_mut() {
            Some(clients) => match clients.get(&client_id) {
                Some(client) => {
                    let client: &Client<R> = client.downcast_ref().unwrap();
                    client.clone()
                }
                None => {
                    let client = R::init_client(device.clone());
                    let any = Box::new(client.clone());
                    clients.insert(client_id, any);
                    client
                }
            },
            _ => unreachable!(),
        }
    }

    fn register_inner<R: RunnerChannel + 'static>(
        key: Key,
        client: Client<R>,
        clients: &mut Option<HashMap<Key, Box<dyn core::any::Any + Send>>>,
    ) {
        if clients.is_none() {
            *clients = Some(HashMap::new());
        }

        if let Some(clients) = clients {
            if clients.contains_key(&key) {
                panic!("Client already created for device {:?}", key);
            }

            clients.insert(key, Box::new(client));
        }
    }
}
