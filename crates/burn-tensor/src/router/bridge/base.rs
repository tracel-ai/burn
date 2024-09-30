use burn_common::stream::StreamId;

use crate::{
    backend::{Backend, DeviceOps},
    repr::{OperationDescription, ReprBackend, TensorDescription},
    router::{Runner, RunnerClient},
    DType, TensorData,
};

pub trait MultiBackendBridge: Send + Sync + 'static {
    type TensorType;
    type Device;
    /// Move `tensor` to the target backend specified `device`.
    fn to_backend(&self, tensor: Self::TensorType, device: &Self::Device) -> Self::TensorType;
}

// TODO: generate this for different number of backends (up to 6?)

/// [`MultiBackendBridge`] tensor type for two backends.
pub enum Handle2<B1: Backend, B2: Backend> {
    FloatHandle1(B1::FloatTensorPrimitive),
    FloatHandle2(B2::FloatTensorPrimitive),
    IntHandle1(B1::IntTensorPrimitive),
    IntHandle2(B2::IntTensorPrimitive),
}

/// [`MultiBackendBridge`] device type for two backends.
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
        Self::Device1(B1::Device::default())
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

/// [`MultiBackendBridge`] client for two backends.
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
