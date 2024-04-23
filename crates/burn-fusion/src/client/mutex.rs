use super::FusionClient;
use crate::{
    stream::{execution::Operation, StreamId},
    FusionBackend, FusionServer, FusionTensor, Handle,
};
use burn_tensor::{
    backend::Backend,
    ops::FloatElem,
    repr::{OperationDescription, TensorDescription, TensorId},
};
use spin::Mutex;
use std::sync::Arc;

/// Use a mutex to communicate with the fusion server.
pub struct MutexFusionClient<B>
where
    B: FusionBackend,
{
    server: Arc<Mutex<FusionServer<B>>>,
    device: B::Device,
}

impl<B> Clone for MutexFusionClient<B>
where
    B: FusionBackend,
{
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
            device: self.device.clone(),
        }
    }
}

impl<B> FusionClient for MutexFusionClient<B>
where
    B: FusionBackend,
{
    type FusionBackend = B;

    fn new(device: B::Device) -> Self {
        Self {
            device: device.clone(),
            server: Arc::new(Mutex::new(FusionServer::new(device))),
        }
    }

    fn register<O: Operation<Self::FusionBackend> + 'static>(
        &self,
        streams: Vec<StreamId>,
        description: OperationDescription,
        operation: O,
    ) {
        self.server
            .lock()
            .register(streams, description, Box::new(operation))
    }

    fn drain(&self) {
        let id = StreamId::current();
        self.server.lock().drain_stream(id);
    }

    fn tensor_uninitialized(&self, shape: Vec<usize>) -> FusionTensor<Self> {
        let id = self.server.lock().create_empty_handle();

        FusionTensor::new(id, shape, self.clone(), StreamId::current())
    }

    fn device(&self) -> &<Self::FusionBackend as Backend>::Device {
        &self.device
    }
    fn register_tensor(
        &self,
        handle: Handle<Self::FusionBackend>,
        shape: Vec<usize>,
        stream: StreamId,
    ) -> FusionTensor<Self> {
        let mut server = self.server.lock();
        let id = server.create_empty_handle();
        server.handles.register_handle(*id.as_ref(), handle);
        core::mem::drop(server);

        FusionTensor::new(id, shape, self.clone(), stream)
    }

    fn read_tensor_float<const D: usize>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<FloatElem<Self::FusionBackend>, D>> {
        self.server.lock().read_float(tensor, stream)
    }

    fn read_tensor_int<const D: usize>(
        &self,
        tensor: TensorDescription,
        id: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<burn_tensor::ops::IntElem<Self::FusionBackend>, D>>
    {
        self.server.lock().read_int(tensor, id)
    }

    fn read_tensor_bool<const D: usize>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<bool, D>> {
        self.server.lock().read_bool(tensor, stream)
    }

    fn change_client_float<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<Self> {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id =
            server_current.change_server_float::<D>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, client, StreamId::current())
    }

    fn change_client_int<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<Self> {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id = server_current.change_server_int::<D>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, client, StreamId::current())
    }

    fn change_client_bool<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<Self> {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id = server_current.change_server_bool::<D>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, client, StreamId::current())
    }

    fn register_orphan(&self, id: &TensorId) {
        self.server.lock().drop_tensor_handle(*id);
    }
}
