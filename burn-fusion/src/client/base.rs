use crate::{
    graph::{GraphExecution, TensorOpsDescription},
    FusedBackend, FusionServer, FusionTensor, TensorDescription, TensorId,
};
use burn_tensor::{
    ops::{FloatElem, IntElem},
    Data, Reader,
};

pub trait FusionClient: Send + Sync + Clone + core::fmt::Debug {
    type FusedBackend: FusedBackend;
    type GraphExecution: GraphExecution<Self::FusedBackend>;

    fn new(server: FusionServer<Self::FusedBackend, Self::GraphExecution>) -> Self;
    fn register(&self, ops: TensorOpsDescription<Self::FusedBackend>);
    fn sync(&self);
    fn device<'a>(&'a self) -> &'a <Self::FusedBackend as FusedBackend>::HandleDevice;
    fn create_empty(&self, shape: Vec<usize>) -> FusionTensor<Self>;
    fn create_float(
        &self,
        values: Vec<FloatElem<Self::FusedBackend>>,
        shape: Vec<usize>,
    ) -> FusionTensor<Self>;
    fn create_int(
        &self,
        values: Vec<IntElem<Self::FusedBackend>>,
        shape: Vec<usize>,
    ) -> FusionTensor<Self>;
    fn create_bool(&self, values: Vec<bool>, shape: Vec<usize>) -> FusionTensor<Self>;
    fn read_float<const D: usize>(
        &self,
        tensor: TensorDescription,
    ) -> Reader<Data<FloatElem<Self::FusedBackend>, D>>;
    fn read_int<const D: usize>(
        &self,
        tensor: TensorDescription,
    ) -> Reader<Data<IntElem<Self::FusedBackend>, D>>;
    fn read_bool<const D: usize>(&self, tensor: TensorDescription) -> Reader<Data<bool, D>>;
    fn change_client_float<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
    ) -> FusionTensor<Self>;
    fn change_client_int<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
    ) -> FusionTensor<Self>;
    fn change_client_bool<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
    ) -> FusionTensor<Self>;
    fn drop_tensor(&self, id: &TensorId);
}
