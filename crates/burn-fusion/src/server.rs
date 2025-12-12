use std::sync::Arc;

use crate::{
    FusionBackend, FusionRuntime,
    stream::{MultiStream, OperationStreams, StreamId, execution::Operation},
};
use burn_backend::{TensorData, backend::ExecutionError};
use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr};

pub struct FusionServer<R: FusionRuntime> {
    streams: MultiStream<R>,
    pub(crate) handles: HandleContainer<R::FusionHandle>,
}

impl<R> FusionServer<R>
where
    R: FusionRuntime,
{
    pub fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: MultiStream::new(device.clone()),
            handles: HandleContainer::new(),
        }
    }

    pub fn register(
        &mut self,
        streams: OperationStreams,
        repr: OperationIr,
        operation: Arc<dyn Operation<R>>,
    ) {
        self.streams
            .register(streams, repr, operation, &mut self.handles)
    }

    pub fn drain_stream(&mut self, id: StreamId) {
        self.streams.drain(&mut self.handles, id)
    }

    pub fn create_empty_handle(&mut self) -> TensorId {
        self.handles.create_tensor_uninit()
    }

    pub fn read_float<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_float = self.handles.get_float_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        B::float_into_data(tensor_float)
    }

    pub fn read_int<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_int = self.handles.get_int_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        B::int_into_data(tensor_int)
    }

    pub fn read_bool<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_bool = self.handles.get_bool_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        B::bool_into_data(tensor_bool)
    }

    pub fn read_quantized<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_q = self.handles.get_quantized_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        B::q_into_data(tensor_q)
    }

    pub fn change_server_float<B>(
        &mut self,
        tensor: &TensorIr,
        stream_tensor: StreamId,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> TensorId
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor_float = self.handles.get_float_tensor::<B>(tensor);
        self.streams.mark_read(stream_tensor, tensor, &self.handles);

        let tensor = B::float_to_device(tensor_float, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_float_tensor::<B>(&id, tensor.clone());

        id
    }

    pub fn resolve_server_float<B>(&mut self, tensor: &TensorIr) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_float_tensor::<B>(tensor)
    }

    pub fn resolve_server_int<B>(&mut self, tensor: &TensorIr) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_int_tensor::<B>(tensor)
    }

    pub fn resolve_server_bool<B>(&mut self, tensor: &TensorIr) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_bool_tensor::<B>(tensor)
    }

    pub fn change_server_int<B>(
        &mut self,
        tensor: &TensorIr,
        stream_tensor: StreamId,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> TensorId
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor_int = self.handles.get_int_tensor::<B>(tensor);
        self.streams.mark_read(stream_tensor, tensor, &self.handles);
        let tensor = B::int_to_device(tensor_int, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_int_tensor::<B>(&id, tensor.clone());

        id
    }

    pub fn change_server_bool<B>(
        &mut self,
        tensor: &TensorIr,
        stream_tensor: StreamId,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> TensorId
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor_bool = self.handles.get_bool_tensor::<B>(tensor);
        self.streams.mark_read(stream_tensor, tensor, &self.handles);
        let tensor = B::bool_to_device(tensor_bool, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_bool_tensor::<B>(&id, tensor.clone());

        id
    }

    pub fn change_server_quantized<B>(
        &mut self,
        tensor: &TensorIr,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> TensorId
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor = self.handles.get_quantized_tensor::<B>(tensor);
        let tensor = B::q_to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_quantized_tensor::<B>(&id, tensor);

        id
    }
}
