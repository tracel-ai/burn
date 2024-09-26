use core::{future::Future, marker::PhantomData};

use burn_common::{stream::StreamId, stub::Mutex};

use crate::{
    backend::{Backend, BackendBridge, DeviceOps},
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

// Runner, Server, Client, Channel
// pub struct DirectChannel<C: MultiRunnerConfig> {
//     /// Runner mappings from ID -> Runner to execute tasks
//     channel: C::Channel,
// }

pub struct BackendRouter<R: MultiBackendRuntime> {
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

impl<R: MultiBackendRuntime> core::fmt::Debug for BackendRouter<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("router"))
    }
}

impl<R: MultiBackendRuntime> Clone for BackendRouter<R> {
    fn clone(&self) -> Self {
        Self { r: PhantomData }
    }
}

impl<R: MultiBackendRuntime> Default for BackendRouter<R> {
    fn default() -> Self {
        Self { r: PhantomData }
    }
}

impl<C: RunnerClient> QTensorPrimitive for RouterTensor<C> {
    fn scheme(&self) -> &crate::quantization::QuantizationScheme {
        todo!()
    }

    fn strategy(&self) -> crate::quantization::QuantizationStrategy {
        todo!()
    }
}

/// Define how to interact with the runner server.
pub trait RunnerClient: Clone + Send + Sync {
    /// Register a new tensor operation to be executed by the (runner) server.
    fn register(&self, op: OperationDescription, info: RouteInfo);
    /// Read the values contained by a tensor.
    fn read_tensor(
        &self,
        tensor: TensorDescription,
        info: RouteInfo,
    ) -> impl Future<Output = TensorData> + Send;
    fn write_tensor(&self, data: TensorData, info: RouteInfo) -> TensorDescription;
    fn empty_tensor(&self, shape: Vec<usize>, dtype: DType, info: RouteInfo) -> TensorDescription;
}

/// Handle precision conversion.
#[derive(Debug)]
pub struct PrecisionBridge {}

impl<R: MultiBackendRuntime> BackendBridge<BackendRouter<R>> for PrecisionBridge {
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

impl<R: MultiBackendRuntime> Backend for BackendRouter<R> {
    type Device = R::Device;

    type FullPrecisionBridge = PrecisionBridge;

    type FloatTensorPrimitive = RouterTensor<R::Client>;

    // TODO: how to set elem types?
    type FloatElem = f32;

    type IntTensorPrimitive = RouterTensor<R::Client>;

    type IntElem = i32;

    type BoolTensorPrimitive = RouterTensor<R::Client>;

    type QuantizedTensorPrimitive = RouterTensor<R::Client>;

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
}

// Runners are used by the channels (e.g., DirectChannel, HttpChannel, etc.)
// Responsible for executing the tensor operations.
#[derive(Clone)]
pub struct Runner<B: ReprBackend> {
    _b: PhantomData<B>,
    context: Mutex<RunnerContext<B>>,
}

impl<B: ReprBackend> RunnerClient for Runner<B> {
    /// Execute a tensor operation.
    fn register(&self, op: crate::repr::OperationDescription, info: RouteInfo) {
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

    async fn read_tensor(&self, tensor: TensorDescription, info: RouteInfo) -> TensorData {
        todo!()
    }

    fn write_tensor(&self, data: TensorData, info: RouteInfo) -> TensorDescription {
        todo!()
    }

    fn empty_tensor(&self, shape: Vec<usize>, dtype: DType, info: RouteInfo) -> TensorDescription {
        todo!()
    }
}

// Defines associated types config for a setup with multiple backend runners.
pub trait MultiBackendRuntime: Clone + Send + Sync + 'static + Sized {
    type Device: DeviceOps;
    type Bridge;
    type Client: RunnerClient;

    /// Get the client for the given device.
    fn client(device: &Device<BackendRouter<Self>>) -> Self::Client;
}

// Allow switching the backend dynamically using a device.
// pub trait BackendChannel: Clone + Send + Sync + 'static + Sized {
//     type Runtime: RunnerRuntime;
//     /// Bridge to transfer tensors between backends.
//     // type Bridge: BackendBridge<BackendRouter<Self>> + 'static;

//     /// Get the runner client for the given device.
//     fn client(device: &Device<Runner<Self>>) -> Client<Self>;
// }

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

impl<B1: ReprBackend, B2: ReprBackend> MultiBackendRuntime for (B1, B2) {
    type Device = MultiDevice2<B1, B2>;

    type Bridge = BackendBridgePlaceholder<B1, B2>; // TODO: replace with MultiBackendBridge to move tensors between backends

    type Client = (Runner<B1>, Runner<B2>);

    fn client(device: &Device<BackendRouter<Self>>) -> Self::Client {
        todo!()
    }
    // type Channel = (Runner<B1>, Runner<B2>);
}

pub struct RouteInfo {
    pub stream: StreamId,
    pub runner_id: usize, // what runner to route the operation to
}

pub enum Handle2<B1: Backend, B2: Backend> {
    Handle1(B1::FloatTensorPrimitive),
    Handle2(B2::FloatTensorPrimitive),
}

// What if instead `move_from` took a TensorRouter, and we could match the device to a specific backend?
//
// pub trait MultiBackendBridge {
//     fn move_from(&self, tensor: RouterTensor) -> RouterTensor<>
// }

// Bridge
pub trait MultiBackendBridge2<B1: Backend, B2: Backend> {
    fn move_from(self, handle: Handle2<B1, B2>) -> Handle2<B1, B2>;
}

struct BackendBridgePlaceholder<B1: Backend, B2: Backend> {
    _b1: PhantomData<B1>,
    _b2: PhantomData<B2>,
}

impl<B1: ReprBackend, B2: ReprBackend> RunnerClient for (Runner<B1>, Runner<B2>) {
    fn register(&self, op: OperationDescription, info: RouteInfo) {
        match info.runner_id {
            0 => self.0.register(op, info),
            1 => self.1.register(op, info),
            _ => panic!(""),
        }
    }

    async fn read_tensor(&self, tensor: TensorDescription, info: RouteInfo) -> TensorData {
        todo!()
    }

    fn write_tensor(&self, data: TensorData, info: RouteInfo) -> TensorDescription {
        todo!()
    }

    fn empty_tensor(&self, shape: Vec<usize>, dtype: DType, info: RouteInfo) -> TensorDescription {
        todo!()
    }
}

// type MyBackend = BackendRouter<DirectChannel<(Cuda, NdArray, Wgpu), ByteBridge>>
// ByteBridge is the default bridge for moving data between backends
// For efficient data movement/transfer, you can implement your own struct
