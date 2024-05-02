use super::FusionClient;
use crate::{
    stream::{execution::Operation, StreamId},
    FusionBackend, FusionDevice, FusionHandle, FusionRuntime, FusionServer, FusionTensor,
};
use burn_tensor::{
    ops::FloatElem,
    repr::{OperationDescription, TensorDescription, TensorId},
    DType,
};
use spin::Mutex;
use std::sync::Arc;

/// Use a mutex to communicate with the fusion server.
pub struct MutexFusionClient<R: FusionRuntime> {
    server: Arc<Mutex<FusionServer<R>>>,
    device: FusionDevice<R>,
}

impl<R> Clone for MutexFusionClient<R>
where
    R: FusionRuntime,
{
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
            device: self.device.clone(),
        }
    }
}

impl<R> FusionClient<R> for MutexFusionClient<R>
where
    R: FusionRuntime<FusionClient = Self>,
{
    fn new(device: FusionDevice<R>) -> Self {
        Self {
            device: device.clone(),
            server: Arc::new(Mutex::new(FusionServer::new(device))),
        }
    }

    fn register<O>(&self, streams: Vec<StreamId>, description: OperationDescription, operation: O)
    where
        O: Operation<R> + 'static,
    {
        self.server
            .lock()
            .register(streams, description, Box::new(operation))
    }

    fn drain(&self) {
        let id = StreamId::current();
        self.server.lock().drain_stream(id);
    }

    fn tensor_uninitialized(&self, shape: Vec<usize>, dtype: DType) -> FusionTensor<R> {
        let id = self.server.lock().create_empty_handle();

        FusionTensor::new(id, shape, dtype, self.clone(), StreamId::current())
    }

    fn device(&self) -> &FusionDevice<R> {
        &self.device
    }

    fn register_tensor(
        &self,
        handle: FusionHandle<R>,
        shape: Vec<usize>,
        stream: StreamId,
        dtype: DType,
    ) -> FusionTensor<R> {
        let mut server = self.server.lock();
        let id = server.create_empty_handle();
        server.handles.register_handle(*id.as_ref(), handle);
        core::mem::drop(server);

        FusionTensor::new(id, shape, dtype, self.clone(), stream)
    }

    fn read_tensor_float<B, const D: usize>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<FloatElem<B>, D>>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_float::<B, D>(tensor, stream)
    }

    fn read_tensor_int<B, const D: usize>(
        &self,
        tensor: TensorDescription,
        id: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<burn_tensor::ops::IntElem<B>, D>>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_int::<B, D>(tensor, id)
    }

    fn read_tensor_bool<B, const D: usize>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> burn_tensor::Reader<burn_tensor::Data<bool, D>>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_bool::<B, D>(tensor, stream)
    }

    fn change_client_float<B, const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id =
            server_current.change_server_float::<B, D>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn change_client_int<B, const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id =
            server_current.change_server_int::<B, D>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn change_client_bool<B, const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id =
            server_current.change_server_bool::<B, D>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn register_orphan(&self, id: &TensorId) {
        self.server.lock().drop_tensor_handle(*id);
    }
}
