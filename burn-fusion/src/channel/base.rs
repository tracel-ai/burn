use crate::{
    graph::{FusedBackend, GraphExecution, TensorOps},
    FusionServer,
};

pub trait FusionChannel<B, G: GraphExecution<B>>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    fn new(server: FusionServer<B, G>) -> Self;
    fn register(&self, ops: TensorOps<B::FloatElem, B::IntElem>);
    fn sync(&self);
}
