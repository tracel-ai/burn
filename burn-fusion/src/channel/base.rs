use std::sync::Arc;

use crate::{
    graph::{FusedBackend, GraphExecution, TensorOps},
    FusionServer, TensorId,
};

pub trait FusionChannel<B, G: GraphExecution<B>>: Send
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    fn new(server: FusionServer<B, G>) -> Self;
    fn register(&self, ops: TensorOps<B::FloatElem, B::IntElem>);
    fn sync(&self);
    fn create(&self, shape: Vec<usize>) -> (B::HandleDevice, Arc<TensorId>);
}
