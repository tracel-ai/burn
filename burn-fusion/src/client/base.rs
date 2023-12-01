use crate::{
    graph::{Ops, TensorOpsDescription},
    FusionBackend, FusionTensor, Handle, TensorDescription, TensorId,
};
use burn_tensor::{
    ops::{FloatElem, IntElem},
    Data, Reader,
};

/// Define how to interact with the fusion server.
pub trait FusionClient: Send + Sync + Clone {
    /// The [fusion backend](FusionBackend) associated type.
    type FusionBackend: FusionBackend;

    /// Create a new client for the given [fusion device](FusionBackend::FusionDevice).
    fn new(device: <Self::FusionBackend as FusionBackend>::FusionDevice) -> Self;
    /// Register a new [tensor operation description](TensorOpsDescription).
    fn register<O: Ops<Self::FusionBackend> + 'static>(
        &self,
        description: TensorOpsDescription,
        ops: O,
    );
    /// Register all lazy computation.
    fn drain_graph(&self);
    /// Get the current device used by all operations handled by this client.
    fn device(&self) -> &<Self::FusionBackend as FusionBackend>::FusionDevice;
    /// Create a new [fusion tensor](FusionTensor), but with no resources allocated to it.
    fn tensor_uninitialized(&self, shape: Vec<usize>) -> FusionTensor<Self>;
    /// Create a tensor with the given handle and shape.
    fn register_tensor(
        &self,
        handle: Handle<Self::FusionBackend>,
        shape: Vec<usize>,
    ) -> FusionTensor<Self>;
    /// Read the values contained by a float tensor.
    fn read_tensor_float<const D: usize>(
        &self,
        tensor: TensorDescription,
    ) -> Reader<Data<FloatElem<Self::FusionBackend>, D>>;
    /// Read the values contained by an int tensor.
    fn read_tensor_int<const D: usize>(
        &self,
        tensor: TensorDescription,
    ) -> Reader<Data<IntElem<Self::FusionBackend>, D>>;
    /// Read the values contained by a bool tensor.
    fn read_tensor_bool<const D: usize>(&self, tensor: TensorDescription) -> Reader<Data<bool, D>>;
    /// Change the client of the given float tensor.
    fn change_client_float<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
    ) -> FusionTensor<Self>;
    /// Change the client of the given int tensor.
    fn change_client_int<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
    ) -> FusionTensor<Self>;
    /// Change the client of the given bool tensor.
    fn change_client_bool<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
    ) -> FusionTensor<Self>;
    /// Drop the tensor with the given [tensor id](TensorId).
    fn register_orphan(&self, id: &TensorId);
}
