use burn_ir::{HandleContainer, OperationIr};
use hashbrown::HashMap;
use smallvec;

use super::{
    StreamId,
    execution::{ExecutionMode, Processor, StreamSegment},
    queue::OperationQueue,
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{FusionRuntime, UnfusedOp};

/// Keep track of multiple concurrent lazy streams of operations.
///
/// Cross-stream tensor sharing is expressed in the IR via [`OperationIr::SharedView`]: when a
/// tensor crosses streams, [`FusionTensor::clone`](crate::FusionTensor::clone) (or `into_ir`)
/// emits a [`SharedView`](OperationIr::SharedView) op that drains the source stream and aliases
/// the underlying handle under a fresh tensor id on the current stream. Every op submitted to a
/// stream after that point operates on tensors that are *local* to that stream — so this struct
/// no longer needs to analyse cross-stream sharing or coordinate deferred drops.
pub struct MultiStream<R: FusionRuntime> {
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
    #[cfg(feature = "memory-checks")]
    memory_checks: super::memory_checks::MemoryChecks,
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            device,
            #[cfg(feature = "memory-checks")]
            memory_checks: super::memory_checks::MemoryChecks::default(),
        }
    }

    /// Register a new tensor operation on the given `stream`.
    ///
    /// [`OperationIr::SharedView`] is intercepted: the source stream is drained and the handle
    /// is aliased eagerly, with nothing enqueued. Every other op is appended to `stream`'s queue
    /// and processed lazily.
    pub(crate) fn register(
        &mut self,
        stream: StreamId,
        repr: OperationIr,
        operation: UnfusedOp<R>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        if let OperationIr::SharedView(view) = &repr {
            self.handle_shared_view(view.clone(), handles);
            #[cfg(feature = "memory-checks")]
            self.memory_checks.check(&self.streams, handles);
            #[cfg(feature = "test-util")]
            crate::inspect::emit_handle_snapshot(stream, handles.handle_ids().copied());
            return;
        }

        // Before enqueueing, drain any *other* streams that still reference this op's input
        // tensors. Without this, lazy ops queued on a peer stream may outlive the source stream's
        // own Drop and try to access a freed handle. (Pre-removal `OperationStreams` did this via
        // `merge_streams_timelines`; with the new invariant cross-stream sharing goes through
        // [`SharedView`], so this only covers the residual "Drop on shared tensor" path.)
        self.drain_peer_streams_for(&repr, stream, handles);

        self.enqueue_operation(stream, repr, operation, handles);

        #[cfg(feature = "memory-checks")]
        self.memory_checks.check(&self.streams, handles);
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(stream, handles.handle_ids().copied());
    }

    /// For each tensor referenced by `repr`, drain any *peer* stream (different from `current`)
    /// whose queue still tracks the tensor. This guarantees that ops registered on the current
    /// stream observe the same ordering they would have under the old per-tensor stream tracking.
    fn drain_peer_streams_for(
        &mut self,
        repr: &OperationIr,
        current: StreamId,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let mut to_drain: smallvec::SmallVec<[StreamId; 4]> = smallvec::SmallVec::new();
        for node in repr.nodes() {
            for (stream_id, stream) in self.streams.iter() {
                if *stream_id != current
                    && stream.queue.variables.contains_key(&node.id)
                    && !to_drain.contains(stream_id)
                {
                    to_drain.push(*stream_id);
                }
            }
        }
        for id in to_drain {
            self.drain(handles, id);
        }
    }

    /// Resolve a [`SharedView`](OperationIr::SharedView) op by draining the source stream and
    /// aliasing the handle under the new tensor id.
    fn handle_shared_view(
        &mut self,
        view: burn_ir::SharedViewOpIr,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        // The source tensor's home stream is the one that owns its handle. Drain it so any
        // pending ops on `src.id` have produced their handle before we alias.
        if let Some(stream) = self.find_owning_stream(view.src.id) {
            self.drain(handles, stream);
        }

        if let Some(handle) = handles.get_handle_ref(&view.src.id).cloned() {
            handles.register_handle(view.out.id, handle);
        }
    }

    /// Find the stream whose queue still tracks the given tensor id.
    fn find_owning_stream(&self, id: burn_ir::TensorId) -> Option<StreamId> {
        for (stream_id, stream) in self.streams.iter() {
            if stream.queue.variables.contains_key(&id) {
                return Some(*stream_id);
            }
        }
        None
    }

    /// Enqueue an operation on the queue for `stream` and run the lazy processor.
    fn enqueue_operation(
        &mut self,
        stream: StreamId,
        repr: OperationIr,
        operation: UnfusedOp<R>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let s = self
            .streams
            .entry(stream)
            .or_insert_with(|| Stream::new(self.device.clone()));
        s.queue.add(repr, operation);

        let len_before = s.queue.global.len();
        s.processor.process(
            Segment::new(&mut s.queue, handles, stream),
            &mut self.optimizations,
            ExecutionMode::Lazy,
        );
        let len_after = s.queue.global.len();
        s.cursor += (len_before - len_after) as u64;
    }

    /// Mark a tensor as read.
    #[allow(unused_variables)]
    pub fn mark_read(
        &mut self,
        id: StreamId,
        ir: &burn_ir::TensorIr,
        handles: &HandleContainer<R::FusionHandle>,
    ) {
        if !matches!(ir.status, burn_ir::TensorStatus::ReadWrite) {
            return;
        };

        let stream = match self.streams.get_mut(&id) {
            Some(val) => val,
            None => return,
        };

        stream.queue.variables.remove(&ir.id);

        if stream.queue.variables.is_empty() {
            self.streams.remove(&id);
        }

        #[cfg(feature = "memory-checks")]
        self.memory_checks.check(&self.streams, handles);
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(id, handles.handle_ids().copied());
    }

    /// Drain a stream.
    pub fn drain(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        id.executes(|| {
            if let Some(stream) = self.streams.get_mut(&id) {
                let num_executed = stream.queue.global.len();
                stream.processor.process(
                    Segment::new(&mut stream.queue, handles, id),
                    &mut self.optimizations,
                    ExecutionMode::Sync,
                );
                stream.cursor += num_executed as u64;
            }
        });
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(id, handles.handle_ids().copied());
    }
}

pub(crate) struct Stream<R: FusionRuntime> {
    pub(crate) queue: OperationQueue<R>,
    processor: Processor<R::Optimization>,
    pub(crate) cursor: u64,
}

#[derive(new)]
struct Segment<'a, R: FusionRuntime> {
    queue: &'a mut OperationQueue<R>,
    handles: &'a mut HandleContainer<R::FusionHandle>,
    id: StreamId,
}

impl<R: FusionRuntime> StreamSegment<R::Optimization> for Segment<'_, R> {
    fn operations(&self) -> &[OperationIr] {
        &self.queue.relative
    }

    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<R::Optimization>) {
        self.queue.execute(id, self.handles, store, self.id)
    }
}

impl<R: FusionRuntime> Stream<R> {
    fn new(device: R::FusionDevice) -> Self {
        Self {
            processor: Processor::new(R::fusers(device)),
            queue: OperationQueue::new(),
            cursor: 0,
        }
    }
}
