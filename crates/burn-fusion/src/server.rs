use crate::{
    FusionBackend, FusionRuntime, UnfusedOp,
    stream::{MultiStream, OperationStreams, StreamId},
};
use burn_backend::{TensorData, backend::ExecutionError};
use burn_ir::{HandleContainer, OperationIr, TensorIr};

pub struct FusionServer<R: FusionRuntime> {
    streams: MultiStream<R>,
    pub(crate) handles: HandleContainer<R::FusionHandle>,
}

impl<R> FusionServer<R>
where
    R: FusionRuntime,
{
    pub fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: MultiStream::new(device.clone()),
            handles: HandleContainer::new(),
        }
    }

    pub fn register(
        &mut self,
        streams: OperationStreams,
        repr: OperationIr,
        operation: UnfusedOp<R>,
    ) {
        self.streams
            .register(streams, repr, operation, &mut self.handles)
    }

    pub fn drain_stream(&mut self, id: StreamId) {
        self.streams.drain(&mut self.handles, id)
    }

    pub fn read_float<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_float = self.handles.get_float_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_float
    }

    pub fn read_int<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_int = self.handles.get_int_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_int
    }

    pub fn read_bool<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_bool = self.handles.get_bool_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_bool
    }

    pub fn read_quantized<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> B::QuantizedTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_q = self.handles.get_quantized_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_q
    }

    pub fn float_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        B::float_into_data(self.read_float::<B>(tensor, id))
    }

    pub fn int_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        B::int_into_data(self.read_int::<B>(tensor, id))
    }

    pub fn bool_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        B::bool_into_data(self.read_bool::<B>(tensor, id))
    }

    pub fn quantized_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        B::q_into_data(self.read_quantized::<B>(tensor, id))
    }

    pub fn resolve_server_float<B>(&mut self, tensor: &TensorIr) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_float_tensor::<B>(tensor)
    }

    pub fn resolve_server_int<B>(&mut self, tensor: &TensorIr) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_int_tensor::<B>(tensor)
    }

    pub fn resolve_server_bool<B>(&mut self, tensor: &TensorIr) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_bool_tensor::<B>(tensor)
    }
}
