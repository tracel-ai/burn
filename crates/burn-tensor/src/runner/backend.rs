use core::{future::Future, marker::PhantomData};

use burn_common::stream::StreamId;

use crate::{
    backend::{Backend, BackendBridge, DeviceOps},
    quantization::QTensorPrimitive,
    repr::{OperationDescription, TensorDescription},
    DType, Device, TensorData,
};

use super::RunnerTensor;

pub struct Runner<B: RunnerBackend> {
    r: PhantomData<B>,
}

pub type Client<B: RunnerBackend> = <B::Runtime as RunnerRuntime>::Client;

pub trait RunnerBackend: Send + Sync + 'static + Sized {
    type Runtime: RunnerRuntime;
    type Bridge: BackendBridge<Runner<Self>> + 'static;

    fn client(device: &Device<Runner<Self>>) -> Client<Self>;
}

pub trait RunnerRuntime {
    type Client: RunnerClient;
    type Device: DeviceOps;
}

impl<B: RunnerBackend> core::fmt::Debug for Runner<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("server"))
    }
}

impl<B: RunnerBackend> Clone for Runner<B> {
    fn clone(&self) -> Self {
        Self { r: PhantomData }
    }
}

impl<B: RunnerBackend> Default for Runner<B> {
    fn default() -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RunnerRuntime> QTensorPrimitive for RunnerTensor<R> {
    fn scheme(&self) -> &crate::quantization::QuantizationScheme {
        todo!()
    }

    fn strategy(&self) -> crate::quantization::QuantizationStrategy {
        todo!()
    }
}

pub trait RunnerClient: Clone + Send + Sync {
    /// Execute an operation.
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

impl<B: RunnerBackend> Backend for Runner<B> {
    type Device = <B::Runtime as RunnerRuntime>::Device;

    type FullPrecisionBridge = B::Bridge;

    type FloatTensorPrimitive<const D: usize> = RunnerTensor<B::Runtime>;

    type FloatElem = f32;

    type IntTensorPrimitive<const D: usize> = RunnerTensor<B::Runtime>;

    type IntElem = i32;

    type BoolTensorPrimitive<const D: usize> = RunnerTensor<B::Runtime>;

    type QuantizedTensorPrimitive<const D: usize> = RunnerTensor<B::Runtime>;

    fn name() -> String {
        todo!()
    }

    fn seed(seed: u64) {
        todo!()
    }
}
