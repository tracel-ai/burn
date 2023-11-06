use crate::{
    graph::{GraphExecution, TensorOps},
    FusedBackend, FusionServer, FusionTensor, TensorDescription,
};
use burn_tensor::{ops::FloatElem, Data, Reader};

pub trait FusionClient: Send + Sync + Clone + core::fmt::Debug {
    type FusedBackend: FusedBackend;
    type GraphExecution: GraphExecution<Self::FusedBackend>;

    fn new(server: FusionServer<Self::FusedBackend, Self::GraphExecution>) -> Self;
    fn register(&self, ops: TensorOps<Self::FusedBackend>);
    fn sync(&self);
    fn device<'a>(&'a self) -> &'a <Self::FusedBackend as FusedBackend>::HandleDevice;
    fn create_empty(&self, shape: Vec<usize>) -> FusionTensor<Self>;
    fn create_float(
        &self,
        values: Vec<FloatElem<Self::FusedBackend>>,
        shape: Vec<usize>,
    ) -> FusionTensor<Self>;
    fn read_float<const D: usize>(
        &self,
        tensor: TensorDescription,
    ) -> Reader<Data<FloatElem<Self::FusedBackend>, D>>;
}
