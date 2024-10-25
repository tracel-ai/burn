use std::future::Future;

use crate::{
    stream::{execution::Operation, StreamId},
    FusionBackend, FusionDevice, FusionHandle, FusionRuntime, FusionTensor, QFusionTensor,
};
use burn_tensor::{
    repr::{OperationDescription, QuantizedTensorDescription, TensorDescription, TensorId},
    DType, TensorData,
};

/// Define how to interact with the fusion server.
pub trait FusionClient<R>: Send + Sync + Clone + Sized
where
    R: FusionRuntime<FusionClient = Self>,
{
    /// Create a new client for the given [device](FusionRuntime::FusionDevice).
    fn new(device: FusionDevice<R>) -> Self;
    /// Register a new [tensor operation description](OperationDescription).
    fn register<O>(&self, streams: Vec<StreamId>, description: OperationDescription, operation: O)
    where
        O: Operation<R> + 'static;
    /// Register all lazy computation.
    fn drain(&self);
    /// Get the current device used by all operations handled by this client.
    fn device(&self) -> &FusionDevice<R>;
    /// Create a new [fusion tensor](FusionTensor), but with no resources allocated to it.
    fn tensor_uninitialized(&self, shape: Vec<usize>, dtype: DType) -> FusionTensor<R>;
    /// Create a tensor with the given handle and shape.
    fn register_tensor(
        &self,
        handle: FusionHandle<R>,
        shape: Vec<usize>,
        stream: StreamId,
        dtype: DType,
    ) -> FusionTensor<R>;
    /// Read the values contained by a float tensor.
    fn read_tensor_float<B>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Read the values contained by an int tensor.
    fn read_tensor_int<B>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Read the values contained by a bool tensor.
    fn read_tensor_bool<B>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Read the values contained by a quantized tensor.
    fn read_tensor_quantized<B>(
        &self,
        tensor: QuantizedTensorDescription,
        streams: Vec<StreamId>,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Change the client of the given float tensor.
    fn change_client_float<B>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Change the client of the given int tensor.
    fn change_client_int<B>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Change the client of the given bool tensor.
    fn change_client_bool<B>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Change the client of the given quantized tensor.
    fn change_client_quantized<B>(
        &self,
        tensor: QuantizedTensorDescription,
        client: Self,
        streams: Vec<StreamId>,
    ) -> QFusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>;
    /// Drop the tensor with the given [tensor id](TensorId).
    fn register_orphan(&self, id: &TensorId);
}
