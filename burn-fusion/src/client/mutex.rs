use super::FusionClient;
use crate::{
    graph::{GraphExecution, TensorOpsDescription},
    FusionBackend, FusionServer, FusionTensor, Handle,
};
use burn_tensor::ops::FloatElem;
use spin::Mutex;
use std::sync::Arc;

/// Use a mutex to communicate with the fusion server.
pub struct MutexFusionClient<B, G>
where
    B: FusionBackend,
    G: GraphExecution<B>,
{
    server: Arc<Mutex<FusionServer<B, G>>>,
    device: B::FusionDevice,
}

impl<B, G> Clone for MutexFusionClient<B, G>
where
    B: FusionBackend,
    G: GraphExecution<B>,
{
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
            device: self.device.clone(),
        }
    }
}

impl<B, G> FusionClient for MutexFusionClient<B, G>
where
    B: FusionBackend,
    G: GraphExecution<B>,
{
    type FusionBackend = B;
    type GraphExecution = G;

    fn new(device: B::FusionDevice) -> Self {
        Self {
            device: device.clone(),
            server: Arc::new(Mutex::new(FusionServer::new(device))),
        }
    }

    fn register<O: crate::graph::Ops<Self::FusionBackend> + 'static>(
        &self,
        description: TensorOpsDescription,
        ops: O,
    ) {
        self.server.lock().register(description, Box::new(ops))
    }

    fn drain_graph(&self) {
        self.server.lock().drain_graph();
    }

    fn tensor_uninitialized(&self, shape: Vec<usize>) -> FusionTensor<Self> {
        let id = self.server.lock().create_empty_handle();

        FusionTensor::new(id, shape, self.clone())
    }

    fn device(&self) -> &<Self::FusionBackend as FusionBackend>::FusionDevice {
        &self.device
    }
    fn register_tensor(
        &self,
        handle: Handle<Self::FusionBackend>,
        shape: Vec<usize>,
    ) -> FusionTensor<Self> {
        let mut server = self.server.lock();
        let id = server.create_empty_handle();
        server.handles.register_handle(id.as_ref().clone(), handle);
        core::mem::drop(server);

        FusionTensor::new(id, shape, self.clone())
    }

    fn read_tensor_float<const D: usize>(
        &self,
        tensor: crate::TensorDescription,
    ) -> burn_tensor::Reader<burn_tensor::Data<FloatElem<Self::FusionBackend>, D>> {
        self.server.lock().read_float(tensor)
    }

    fn read_tensor_int<const D: usize>(
        &self,
        tensor: crate::TensorDescription,
    ) -> burn_tensor::Reader<burn_tensor::Data<burn_tensor::ops::IntElem<Self::FusionBackend>, D>>
    {
        self.server.lock().read_int(tensor)
    }

    fn read_tensor_bool<const D: usize>(
        &self,
        tensor: crate::TensorDescription,
    ) -> burn_tensor::Reader<burn_tensor::Data<bool, D>> {
        self.server.lock().read_bool(tensor)
    }

    fn change_client_float<const D: usize>(
        &self,
        tensor: crate::TensorDescription,
        client: Self,
    ) -> FusionTensor<Self> {
        let device = client.device.clone().into();

        let mut other_server = client.server.lock();

        let id = self
            .server
            .lock()
            .change_server_float::<D>(&tensor, &device, &mut other_server);

        core::mem::drop(other_server);

        FusionTensor::new(id, tensor.shape, client)
    }
    fn change_client_int<const D: usize>(
        &self,
        tensor: crate::TensorDescription,
        client: Self,
    ) -> FusionTensor<Self> {
        let device = client.device.clone().into();

        let mut other_server = client.server.lock();

        let id = self
            .server
            .lock()
            .change_server_int::<D>(&tensor, &device, &mut other_server);

        core::mem::drop(other_server);

        FusionTensor::new(id, tensor.shape, client)
    }

    fn change_client_bool<const D: usize>(
        &self,
        tensor: crate::TensorDescription,
        client: Self,
    ) -> FusionTensor<Self> {
        let device = client.device.clone().into();

        let mut other_server = client.server.lock();

        let id = self
            .server
            .lock()
            .change_server_bool::<D>(&tensor, &device, &mut other_server);

        core::mem::drop(other_server);

        FusionTensor::new(id, tensor.shape, client)
    }

    fn register_orphan(&self, id: &crate::TensorId) {
        self.server.lock().drop_tensor_handle(id.clone());
    }
}
