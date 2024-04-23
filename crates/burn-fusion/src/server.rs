use crate::{
    stream::{execution::Operation, MultiStream, StreamId},
    FusionBackend,
};
use burn_tensor::{
    ops::{FloatElem, IntElem},
    repr::{HandleContainer, OperationDescription, TensorDescription, TensorId},
};
use std::sync::Arc;

pub struct FusionServer<B>
where
    B: FusionBackend,
{
    streams: MultiStream<B>,
    pub(crate) handles: HandleContainer<B>,
    pub device: B::Device,
}

impl<B> FusionServer<B>
where
    B: FusionBackend,
{
    pub fn new(device: B::Device) -> Self {
        Self {
            streams: MultiStream::new(device.clone()),
            handles: HandleContainer::new(device.clone()),
            device,
        }
    }

    pub fn register(
        &mut self,
        streams: Vec<StreamId>,
        desc: OperationDescription,
        operation: Box<dyn Operation<B>>,
    ) {
        self.streams
            .register(streams, desc, operation, &mut self.handles)
    }

    pub fn drain_stream(&mut self, id: StreamId) {
        self.streams.drain(&mut self.handles, id)
    }

    pub fn create_empty_handle(&mut self) -> Arc<TensorId> {
        self.handles.create_tensor_uninit()
    }

    pub fn read_float<const D: usize>(
        &mut self,
        tensor: TensorDescription,
        id: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<FloatElem<B>, D>> {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);

        let tensor = self.handles.get_float_tensor(&tensor);
        B::float_into_data(tensor)
    }

    pub fn read_int<const D: usize>(
        &mut self,
        tensor: TensorDescription,
        id: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<IntElem<B>, D>> {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);

        let tensor = self.handles.get_int_tensor(&tensor);
        B::int_into_data(tensor)
    }

    pub fn read_bool<const D: usize>(
        &mut self,
        tensor: TensorDescription,
        id: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<bool, D>> {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);

        let tensor = self.handles.get_bool_tensor(&tensor);
        B::bool_into_data(tensor)
    }

    pub fn change_server_float<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
        device: &B::Device,
        server_device: &mut Self,
    ) -> Arc<TensorId> {
        let tensor = self.handles.get_float_tensor::<D>(tensor);
        let tensor = B::float_to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_float_tensor(&id, tensor.clone());

        id
    }
    pub fn change_server_int<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
        device: &B::Device,
        server_device: &mut Self,
    ) -> Arc<TensorId> {
        let tensor = self.handles.get_int_tensor::<D>(tensor);
        let tensor = B::int_to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_int_tensor(&id, tensor.clone());

        id
    }
    pub fn change_server_bool<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
        device: &B::Device,
        server_device: &mut Self,
    ) -> Arc<TensorId> {
        let tensor = self.handles.get_bool_tensor::<D>(tensor);
        let tensor = B::bool_to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_bool_tensor(&id, tensor.clone());

        id
    }

    pub fn drop_tensor_handle(&mut self, id: TensorId) {
        self.handles.handles_orphan.push(id);
    }
}
