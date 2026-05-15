use super::{
    StreamId,
    execution::{ExecutionMode, Processor, StreamSegment},
    queue::OperationQueue,
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{FusionRuntime, UnfusedOp};
use burn_ir::{HandleContainer, OperationIr, TensorId};
use hashbrown::{HashMap, HashSet};

/// Keep track of multiple concurrent lazy streams of operations.
///
/// Cross-stream tensor sharing is handled out-of-band by [`Self::tag_shared_view`]: when a
/// tensor crosses streams, [`FusionTensor::clone`](crate::FusionTensor::clone) (or `into_ir`)
/// drains the source stream once and aliases its backing handle under a fresh tensor id on the
/// current stream. No new [`OperationIr`] variant is involved — every op submitted to a stream
/// after that point operates on tensors that are *local* to that stream, so this struct no
/// longer needs to analyse cross-stream sharing or coordinate deferred drops.
pub struct MultiStream<R: FusionRuntime> {
    resolved: HashSet<TensorId>,
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
    #[cfg(feature = "memory-checks")]
    memory_checks: super::memory_checks::MemoryChecks,
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            resolved: HashSet::new(),
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            device,
            #[cfg(feature = "memory-checks")]
            memory_checks: super::memory_checks::MemoryChecks::default(),
        }
    }

    /// Register a new tensor operation on the given `stream`.
    pub(crate) fn register(
        &mut self,
        stream: StreamId,
        repr: OperationIr,
        operation: UnfusedOp<R>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        self.enqueue_operation(stream, repr, operation, handles);

        #[cfg(feature = "memory-checks")]
        self.memory_checks.check(&self.streams, handles);
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(stream, handles.handle_ids().copied());
    }

    pub(crate) fn tag_shared_view(
        &mut self,
        src_stream: StreamId,
        src: TensorId,
        dst: TensorId,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        if !self.resolved.contains(&src) {
            self.resolved.insert(src);
            // Only drain when it's the first time a tensor is tagged as being shared.
            // After that the handle is up to date and can be used by other streams.
            self.drain(handles, src_stream);
        }

        if let Some(handle) = handles.get_handle_ref(&src) {
            handles.register_handle(dst, handle.clone());
        }
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
