use core::marker::PhantomData;

use crate::{
    backend::{Backend, BackendBridge, DeviceOps},
    quantization::QTensorPrimitive,
    repr::{OperationDescription, TensorDescription},
};

pub struct Server<B: ServerBackend> {
    r: PhantomData<B>,
}

pub trait ServerBackend: Send + Sync + 'static {
    type Runtime: ServerRuntime;
}

pub trait ServerRuntime {
    type Client: ServerClient;
    type Device: DeviceOps;
    type Bridge;
}

pub struct ServerTensor<R: ServerRuntime> {
    desc: TensorDescription,
    client: R::Client,
}

impl<B: ServerBackend> core::fmt::Debug for Server<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("server"))
    }
}

impl<R: ServerRuntime> core::fmt::Debug for ServerTensor<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("tensor"))
    }
}

impl<R: ServerRuntime> Clone for ServerTensor<R> {
    fn clone(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            client: self.client.clone(),
        }
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
    fn execute(&self, op: OperationDescription);
}

impl<B: ServerBackend> Backend for Server<B>
where
    <B::Runtime as ServerRuntime>::Bridge: BackendBridge<Self> + 'static,
{
    type Device = <B::Runtime as ServerRuntime>::Device;

    type FullPrecisionBridge = <B::Runtime as ServerRuntime>::Bridge;

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
