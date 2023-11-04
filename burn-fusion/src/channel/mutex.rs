use std::sync::Arc;

use super::FusionChannel;
use crate::{
    graph::{FusedBackend, GraphExecution, TensorOps},
    FusionServer, TensorId,
};
use spin::Mutex;

pub struct MutexFusionChannel<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    server: Mutex<FusionServer<B, G>>,
}

impl<B: FusedBackend, G: GraphExecution<B>> FusionChannel<B, G> for MutexFusionChannel<B, G> {
    fn new(server: FusionServer<B, G>) -> Self {
        Self {
            server: Mutex::new(server),
        }
    }
    fn register(&self, ops: TensorOps<B::FloatElem, B::IntElem>) {
        self.server.lock().register(ops);
    }

    fn sync(&self) {
        self.server.lock().sync();
    }
    fn create(&self, shape: Vec<usize>) -> (B::HandleDevice, Arc<TensorId>) {
        self.server.lock().create(shape)
    }
}
