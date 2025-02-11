use crate::{
    stream::{execution::Operation, MultiStream, StreamId},
    FusionBackend, FusionRuntime,
};
use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr};
use std::{future::Future, sync::Arc};

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
        streams: Vec<StreamId>,
        repr: OperationIr,
        operation: Box<dyn Operation<R>>,
    ) {
        self.streams
            .register(streams, repr, operation, &mut self.handles)
    }

    pub fn drain_stream(&mut self, id: StreamId) {
        self.streams.drain(&mut self.handles, id)
    }

    pub fn create_empty_handle(&mut self) -> Arc<TensorId> {
        self.handles.create_tensor_uninit()
    }

    pub fn read_float<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = burn_tensor::TensorData> + 'static
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);

        let tensor = self.handles.get_float_tensor::<B>(&tensor);
        B::float_into_data(tensor)
    }

    pub fn read_int<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = burn_tensor::TensorData> + 'static
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);

        let tensor = self.handles.get_int_tensor::<B>(&tensor);
        B::int_into_data(tensor)
    }

    pub fn read_bool<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = burn_tensor::TensorData> + 'static
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);

        let tensor = self.handles.get_bool_tensor::<B>(&tensor);
        B::bool_into_data(tensor)
    }

    pub fn read_quantized<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = burn_tensor::TensorData> + 'static
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);

        let tensor = self.handles.get_quantized_tensor::<B>(&tensor);
        B::q_into_data(tensor)
    }

    pub fn change_server_float<B>(
        &mut self,
        tensor: &TensorIr,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> Arc<TensorId>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor = self.handles.get_float_tensor::<B>(tensor);
        let tensor = B::float_to_device(tensor, device);
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
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> Arc<TensorId>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor = self.handles.get_int_tensor::<B>(tensor);
        let tensor = B::int_to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_int_tensor::<B>(&id, tensor.clone());

        id
    }

    pub fn change_server_bool<B>(
        &mut self,
        tensor: &TensorIr,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> Arc<TensorId>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor = self.handles.get_bool_tensor::<B>(tensor);
        let tensor = B::bool_to_device(tensor, device);
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
    ) -> Arc<TensorId>
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

    pub fn drop_tensor_handle(&mut self, id: TensorId) {
        self.handles.handles_orphan.push(id);
    }
}
