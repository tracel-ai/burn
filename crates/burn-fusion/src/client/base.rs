use std::future::Future;

use crate::{
    FusionBackend, FusionDevice, FusionHandle, FusionRuntime, FusionTensor,
    stream::{OperationStreams, StreamId, execution::Operation},
};
use burn_ir::{OperationIr, TensorIr};
use burn_tensor::{DType, Shape, TensorData};

/// Define how to interact with the fusion server.
pub trait FusionClient<R>: Send + Sync + Clone + Sized
where
    R: FusionRuntime<FusionClient = Self>,
{
    /// Create a new client for the given [device](FusionRuntime::FusionDevice).
    fn new(device: FusionDevice<R>) -> Self;
    /// Register a new [tensor operation intermediate representation](OperationIr).
    fn register<O>(&self, streams: OperationStreams, repr: OperationIr, operation: O)
    where
        O: Operation<R> + 'static;
    /// Register all lazy computation.
    fn drain(&self);
    /// Get the current device used by all operations handled by this client.
    fn device(&self) -> &FusionDevice<R>;
    /// Create a new [fusion tensor](FusionTensor), but with no resources allocated to it.
    fn tensor_uninitialized(&self, shape: Shape, dtype: DType) -> FusionTensor<R>;
    /// Create a tensor with the given handle and shape.
    fn register_tensor(
        &self,
        handle: FusionHandle<R>,
        shape: Shape,
        stream: StreamId,
        dtype: DType,
    ) -> FusionTensor<R>;
    /// Read the values contained by a float tensor.
    fn read_tensor_float<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Read the values contained by an int tensor.
    fn read_tensor_int<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Read the values contained by a bool tensor.
    fn read_tensor_bool<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Read the values contained by a quantized tensor.
    fn read_tensor_quantized<B>(
        self,
        tensor: TensorIr,
        streams: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Resolve the given float tensor to a primitive tensor.
    fn resolve_tensor_float<B>(&self, tensor: FusionTensor<R>) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Resolve the given int tensor to a primitive tensor.
    fn resolve_tensor_int<B>(&self, tensor: FusionTensor<R>) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Resolve the given bool tensor to a primitive tensor.
    fn resolve_tensor_bool<B>(&self, tensor: FusionTensor<R>) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Change the client of the given float tensor.
    fn change_client_float<B>(
        &self,
        tensor: TensorIr,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Change the client of the given int tensor.
    fn change_client_int<B>(
        &self,
        tensor: TensorIr,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Change the client of the given bool tensor.
    fn change_client_bool<B>(
        &self,
        tensor: TensorIr,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Change the client of the given quantized tensor.
    fn change_client_quantized<B>(
        &self,
        tensor: TensorIr,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>;
}
