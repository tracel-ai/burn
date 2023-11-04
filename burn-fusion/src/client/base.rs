use crate::{
    graph::{FusedBackend, GraphExecution, TensorOps},
    FusionServer, FusionTensor,
};
use burn_tensor::backend::Backend;

pub trait FusionClient: Send + Clone {
    type FusedBackend: FusedBackend;
    type GraphExecution: GraphExecution<Self::FusedBackend>;

    fn new(server: FusionServer<Self::FusedBackend, Self::GraphExecution>) -> Self;
    fn register(
        &self,
        ops: TensorOps<
            <Self::FusedBackend as Backend>::FloatElem,
            <Self::FusedBackend as Backend>::IntElem,
        >,
    );
    fn sync(&self);
    fn empty(&self, shape: Vec<usize>) -> FusionTensor<Self>;
    fn device<'a>(&'a self) -> &'a <Self::FusedBackend as FusedBackend>::HandleDevice;
}
