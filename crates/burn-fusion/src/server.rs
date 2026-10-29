use crate::{
    FusionBackend, FusionRuntime, UnfusedOp,
    stream::{MultiStream, ReadPlan, StreamId},
};
use burn_backend::{TensorData, backend::ExecutionError};
use burn_ir::{HandleContainer, OperationIr, TensorError, TensorId, TensorIr, TensorStatus};

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

    pub fn register(&mut self, stream: StreamId, repr: OperationIr, operation: UnfusedOp<R>) {
        self.streams
            .register(stream, repr, operation, &mut self.handles)
    }

    /// Free a tensor dropped on a thread other than its home stream.
    ///
    /// A foreign drop must neither enqueue into the pending segment (the block DAG could reorder
    /// the free ahead of a pending read) nor cut it by draining (see
    /// [`ReadPlan`](crate::stream::ReadPlan)), so it never touches the queue.
    pub fn foreign_drop(&mut self, stream: StreamId, ir: TensorIr) {
        self.streams.foreign_drop(stream, ir, &mut self.handles);
    }

    pub fn tag_shared_view(&mut self, src_stream: StreamId, src: TensorId, dst: TensorId) {
        self.streams
            .tag_shared_view(src_stream, src, dst, &mut self.handles)
    }

    pub fn drain_stream(&mut self, id: StreamId) {
        self.streams.drain(&mut self.handles, id)
    }

    /// Ready `id`'s stream for reading `tensor` and return the IR the handle
    /// lookup must use.
    ///
    /// A `DeferFree` read goes through a `ReadOnly` view; its free runs at
    /// the stream's next execution boundary. See [`ReadPlan`].
    fn prepare_read(&mut self, tensor: &TensorIr, id: StreamId) -> TensorIr {
        match self.streams.read_plan(id, tensor, &self.handles) {
            ReadPlan::Drain => {
                self.drain_stream(id);
                tensor.clone()
            }
            ReadPlan::Direct => tensor.clone(),
            ReadPlan::DeferFree => {
                self.streams.defer_free(id, tensor.clone());
                TensorIr {
                    status: TensorStatus::ReadOnly,
                    ..tensor.clone()
                }
            }
        }
    }

    pub fn read_float<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> Result<B::FloatTensorPrimitive, TensorError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The underlying backend can still be async.
        let tensor = self.prepare_read(&tensor, id);
        // Asked after the drain, which is what runs the work that was going to
        // write this tensor — and so what records the failure claiming it if
        // that work never got there.
        let tensor_float = match self.handles.take_error(&tensor) {
            Some(error) => Err(error),
            None => Ok(self.handles.get_float_tensor::<B>(&tensor)),
        };
        // Stream bookkeeping either way: the read happened, whatever it found.
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_float
    }

    pub fn read_int<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> Result<B::IntTensorPrimitive, TensorError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The underlying backend can still be async.
        let tensor = self.prepare_read(&tensor, id);
        // Asked after the drain, which is what runs the work that was going to
        // write this tensor — and so what records the failure claiming it if
        // that work never got there.
        let tensor_int = match self.handles.take_error(&tensor) {
            Some(error) => Err(error),
            None => Ok(self.handles.get_int_tensor::<B>(&tensor)),
        };
        // Stream bookkeeping either way: the read happened, whatever it found.
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_int
    }

    pub fn read_bool<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> Result<B::BoolTensorPrimitive, TensorError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The underlying backend can still be async.
        let tensor = self.prepare_read(&tensor, id);
        // Asked after the drain, which is what runs the work that was going to
        // write this tensor — and so what records the failure claiming it if
        // that work never got there.
        let tensor_bool = match self.handles.take_error(&tensor) {
            Some(error) => Err(error),
            None => Ok(self.handles.get_bool_tensor::<B>(&tensor)),
        };
        // Stream bookkeeping either way: the read happened, whatever it found.
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_bool
    }

    pub fn read_quantized<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> Result<B::QuantizedTensorPrimitive, TensorError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The underlying backend can still be async.
        let tensor = self.prepare_read(&tensor, id);
        // Asked after the drain, which is what runs the work that was going to
        // write this tensor — and so what records the failure claiming it if
        // that work never got there.
        let tensor_q = match self.handles.take_error(&tensor) {
            Some(error) => Err(error),
            None => Ok(self.handles.get_quantized_tensor::<B>(&tensor)),
        };
        // Stream bookkeeping either way: the read happened, whatever it found.
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
        // The claim check is synchronous and belongs with the drain that
        // records it, so it runs here on the server thread; only the backend's
        // read is awaited.
        let read = self.read_float::<B>(tensor, id).map(B::float_into_data);

        async move {
            match read {
                Ok(data) => data.await,
                Err(error) => Err(error.into()),
            }
        }
    }

    pub fn int_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The claim check is synchronous and belongs with the drain that
        // records it, so it runs here on the server thread; only the backend's
        // read is awaited.
        let read = self.read_int::<B>(tensor, id).map(B::int_into_data);

        async move {
            match read {
                Ok(data) => data.await,
                Err(error) => Err(error.into()),
            }
        }
    }

    pub fn bool_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The claim check is synchronous and belongs with the drain that
        // records it, so it runs here on the server thread; only the backend's
        // read is awaited.
        let read = self.read_bool::<B>(tensor, id).map(B::bool_into_data);

        async move {
            match read {
                Ok(data) => data.await,
                Err(error) => Err(error.into()),
            }
        }
    }

    pub fn quantized_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The claim check is synchronous and belongs with the drain that
        // records it, so it runs here on the server thread; only the backend's
        // read is awaited.
        let read = self.read_quantized::<B>(tensor, id).map(B::q_into_data);

        async move {
            match read {
                Ok(data) => data.await,
                Err(error) => Err(error.into()),
            }
        }
    }

    pub fn resolve_server_float<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> Result<B::FloatTensorPrimitive, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.read_float::<B>(tensor, id)
            .map_err(ExecutionError::from)
    }

    pub fn resolve_server_int<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> Result<B::IntTensorPrimitive, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.read_int::<B>(tensor, id).map_err(ExecutionError::from)
    }

    pub fn resolve_server_bool<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> Result<B::BoolTensorPrimitive, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.read_bool::<B>(tensor, id)
            .map_err(ExecutionError::from)
    }
}
