use super::FusionClient;
use crate::{
    graph::{GraphExecution, TensorOps},
    FusedBackend, FusionServer, FusionTensor,
};
use burn_tensor::ops::FloatElem;
use spin::Mutex;
use std::sync::Arc;

pub struct MutexFusionClient<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    server: Arc<Mutex<FusionServer<B, G>>>,
    device: B::HandleDevice,
}

impl<B, G> Clone for MutexFusionClient<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
            device: self.device.clone(),
        }
    }
}

impl<B, G> core::fmt::Debug for MutexFusionClient<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<B, G> FusionClient for MutexFusionClient<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    type FusedBackend = B;
    type GraphExecution = G;

    fn new(server: FusionServer<B, G>) -> Self {
        Self {
            device: server.device.clone(),
            server: Arc::new(Mutex::new(server)),
        }
    }

    fn register(&self, ops: TensorOps<B>) {
        self.server.lock().register(ops);
    }

    fn sync(&self) {
        self.server.lock().sync();
    }
    fn create_empty(&self, shape: Vec<usize>) -> FusionTensor<Self> {
        let id = self.server.lock().create_empty_handle();

        FusionTensor {
            id,
            shape,
            client: self.clone(),
        }
    }

    fn device<'a>(&'a self) -> &'a <Self::FusedBackend as FusedBackend>::HandleDevice {
        &self.device
    }

    fn read_float<const D: usize>(
        &self,
        tensor: crate::TensorDescription,
    ) -> burn_tensor::Reader<burn_tensor::Data<FloatElem<Self::FusedBackend>, D>> {
        self.server.lock().read_float(tensor)
    }

    fn create_float(
        &self,
        values: Vec<FloatElem<Self::FusedBackend>>,
        shape: Vec<usize>,
    ) -> FusionTensor<Self> {
        let id = self.server.lock().create_float_handle(values);

        FusionTensor {
            id,
            shape,
            client: self.clone(),
        }
    }
}
