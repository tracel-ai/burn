use crate::{
    stream::{execution::Operation, StreamId},
    FusionBackend, FusionTensor, Handle,
};
use burn_tensor::{
    backend::Backend,
    ops::{FloatElem, IntElem},
    repr::{OperationDescription, TensorDescription, TensorId},
    Data, Device, Reader,
};

/// Define how to interact with the fusion server.
pub trait FusionClient: Send + Sync + Clone {
    /// The [fusion backend](FusionBackend) associated type.
    type FusionBackend: FusionBackend;

    /// Create a new client for the given [device](Backend::Device).
    fn new(device: Device<Self::FusionBackend>) -> Self;
    /// Register a new [tensor operation description](OperationDescription).
    fn register<O: Operation<Self::FusionBackend> + 'static>(
        &self,
        streams: Vec<StreamId>,
        description: OperationDescription,
        operation: O,
    );
    /// Register all lazy computation.
    fn drain(&self);
    /// Get the current device used by all operations handled by this client.
    fn device(&self) -> &<Self::FusionBackend as Backend>::Device;
    /// Create a new [fusion tensor](FusionTensor), but with no resources allocated to it.
    fn tensor_uninitialized(&self, shape: Vec<usize>) -> FusionTensor<Self>;
    /// Create a tensor with the given handle and shape.
    fn register_tensor(
        &self,
        handle: Handle<Self::FusionBackend>,
        shape: Vec<usize>,
        stream: StreamId,
    ) -> FusionTensor<Self>;
    /// Read the values contained by a float tensor.
    fn read_tensor_float<const D: usize>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> Reader<Data<FloatElem<Self::FusionBackend>, D>>;
    /// Read the values contained by an int tensor.
    fn read_tensor_int<const D: usize>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> Reader<Data<IntElem<Self::FusionBackend>, D>>;
    /// Read the values contained by a bool tensor.
    fn read_tensor_bool<const D: usize>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> Reader<Data<bool, D>>;
    /// Change the client of the given float tensor.
    fn change_client_float<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<Self>;
    /// Change the client of the given int tensor.
    fn change_client_int<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<Self>;
    /// Change the client of the given bool tensor.
    fn change_client_bool<const D: usize>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<Self>;
    /// Drop the tensor with the given [tensor id](TensorId).
    fn register_orphan(&self, id: &TensorId);
}
