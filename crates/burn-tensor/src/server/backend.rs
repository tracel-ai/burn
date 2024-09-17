use core::{future::Future, marker::PhantomData};

use burn_common::stream::StreamId;

use crate::{
    backend::{Backend, BackendBridge, DeviceOps},
    quantization::QTensorPrimitive,
    repr::{OperationDescription, TensorDescription},
    TensorData,
};

use super::ServerTensor;

pub struct Server<B: ServerBackend> {
    r: PhantomData<B>,
}

pub trait ServerBackend: Send + Sync + 'static + Sized {
    type Runtime: ServerRuntime;
    type Bridge: BackendBridge<Server<Self>> + 'static;
}

pub trait ServerRuntime {
    type Client: ServerClient;
    type Device: DeviceOps;
}

impl<B: ServerBackend> core::fmt::Debug for Server<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("server"))
    }
}

impl<B: ServerBackend> Clone for Server<B> {
    fn clone(&self) -> Self {
        Self { r: PhantomData }
    }
}

impl<B: ServerBackend> Default for Server<B> {
    fn default() -> Self {
        Self { r: PhantomData }
    }
}

impl<R: ServerRuntime> QTensorPrimitive for ServerTensor<R> {
    fn scheme(&self) -> &crate::quantization::QuantizationScheme {
        todo!()
    }

    fn strategy(&self) -> crate::quantization::QuantizationStrategy {
        todo!()
    }
}

pub trait ServerClient: Clone + Send + Sync {
    /// Execute an operation.
    fn execute(&self, op: OperationDescription);
    /// Read the values contained by a tensor.
    fn read_tensor(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send;
}

impl<B: ServerBackend> Backend for Server<B> {
    type Device = <B::Runtime as ServerRuntime>::Device;

    type FullPrecisionBridge = B::Bridge;

    type FloatTensorPrimitive<const D: usize> = ServerTensor<B::Runtime>;

    type FloatElem = f32;

    type IntTensorPrimitive<const D: usize> = ServerTensor<B::Runtime>;

    type IntElem = i32;

    type BoolTensorPrimitive<const D: usize> = ServerTensor<B::Runtime>;

    type QuantizedTensorPrimitive<const D: usize> = ServerTensor<B::Runtime>;

    fn name() -> String {
        todo!()
    }

    fn seed(seed: u64) {
        todo!()
    }
}
