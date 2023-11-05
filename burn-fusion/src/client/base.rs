use crate::{
    graph::{FusedBackend, GraphExecution, TensorOps},
    FusionServer, FusionTensor,
};

pub trait FusionClient: Send + Clone {
    type FusedBackend: FusedBackend;
    type GraphExecution: GraphExecution<Self::FusedBackend>;

    fn new(server: FusionServer<Self::FusedBackend, Self::GraphExecution>) -> Self;
    fn register(&self, ops: TensorOps<Self::FusedBackend>);
    fn sync(&self);
    fn empty(&self, shape: Vec<usize>) -> FusionTensor<Self>;
    fn device<'a>(&'a self) -> &'a <Self::FusedBackend as FusedBackend>::HandleDevice;
}
