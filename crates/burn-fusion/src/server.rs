use std::sync::Arc;

use crate::{
    FusionBackend, FusionRuntime, UnfusedOp,
    stream::{MultiStream, ReadPlan, StreamId},
};
use burn_backend::{TensorData, backend::ExecutionError};
use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr, TensorStatus};
use burn_std::{CommunicationId, sync::RwLock};
use hashbrown::HashSet;

pub(crate) struct FusionUtilities {
    // Used in client using a downcast.
    #[allow(dead_code)]
    pub(crate) initialized_comms: RwLock<HashSet<CommunicationId>>,
}

pub struct FusionServer<R: FusionRuntime> {
    streams: MultiStream<R>,
    pub(crate) handles: HandleContainer<R::FusionHandle>,
    pub(crate) utilities: Arc<FusionUtilities>,
}

impl<R> FusionServer<R>
where
    R: FusionRuntime,
{
    pub fn new(device: R::FusionDevice, utilities: FusionUtilities) -> Self {
        Self {
            streams: MultiStream::new(device.clone()),
            handles: HandleContainer::new(),
            utilities: Arc::new(utilities),
        }
    }

    pub fn register(&mut self, stream: StreamId, repr: OperationIr, operation: UnfusedOp<R>) {
        self.streams
            .register(stream, repr, operation, &mut self.handles)
    }

    /// Register a `Drop` that originates from a thread other than the tensor's home stream.
    ///
    /// A same-thread drop is naturally ordered after that thread's last use of the id, so the
    /// fusion lifetime analysis frees it safely. A *foreign* drop arrives at a nondeterministic
    /// point relative to the home thread's own registrations, so it must neither enter the
    /// still-building fused segment (the block DAG (#5135) could reorder the free ahead of a
    /// pending read and let the buffer be reused while an in-flight kernel still reads it) nor
    /// cut that segment by draining — the cut point would be set by cross-thread timing, and the
    /// same op sequence would compile different fused blocks run to run (see
    /// [`ReadPlan`](crate::stream::ReadPlan)).
    ///
    /// A materialized tensor's drop therefore bypasses the queue entirely: freed immediately, or
    /// at the next execution boundary if pending ops still reference the tensor. Only a tensor
    /// whose producer is itself still pending falls back to drain-then-enqueue, since only the
    /// queue can order that drop after the producer.
    pub fn register_foreign_drop(
        &mut self,
        stream: StreamId,
        ir: TensorIr,
        operation: UnfusedOp<R>,
    ) {
        if self.streams.foreign_drop(stream, ir.clone(), &mut self.handles) {
            return;
        }
        self.streams.drain(&mut self.handles, stream);
        self.streams
            .register(stream, OperationIr::Drop(ir), operation, &mut self.handles);
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
    /// Pending operations are executed only when the read requires it: the
    /// tensor's handle does not exist yet. A read of a materialized tensor
    /// leaves the queue untouched — draining here would cut the
    /// still-building fused composition at a point set by cross-thread
    /// timing (see [`ReadPlan`]) — and a materialized *last use* whose
    /// tensor pending operations still reference reads through a `ReadOnly`
    /// view, its free deferred to the stream's next execution boundary.
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

    pub fn read_float<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The underlying backend can still be async.
        let tensor = self.prepare_read(&tensor, id);
        let tensor_float = self.handles.get_float_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_float
    }

    pub fn read_int<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The underlying backend can still be async.
        let tensor = self.prepare_read(&tensor, id);
        let tensor_int = self.handles.get_int_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_int
    }

    pub fn read_bool<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // The underlying backend can still be async.
        let tensor = self.prepare_read(&tensor, id);
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
        // The underlying backend can still be async.
        let tensor = self.prepare_read(&tensor, id);
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
