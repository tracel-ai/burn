use burn_ir::{HandleContainer, OperationIr, TensorIr};

use super::{
    execution::{ExecutionMode, Operation, Processor, StreamSegment},
    store::{ExecutionPlanId, ExecutionPlanStore},
    OperationQueue, StreamId,
};
use crate::FusionRuntime;
use std::collections::HashMap;

/// Keep track of multiple concurrent streams of operations.
pub struct MultiStream<R: FusionRuntime> {
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            device,
        }
    }

    /// Register a new tensor operation.
    pub(crate) fn register(
        &mut self,
        streams: Vec<StreamId>,
        repr: OperationIr,
        operation: Box<dyn Operation<R>>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let id = self.resolve_streams(streams, handles, &repr);

        let stream = match self.streams.get_mut(&id) {
            Some(stream) => stream,
            None => {
                let stream = Stream::new(self.device.clone());
                self.streams.insert(id, stream);
                self.streams
                    .get_mut(&id)
                    .expect("Just added, so should be included in the hashmap.")
            }
        };

        stream.queue.add(repr, operation);

        let size_before = stream.queue.len();
        stream.processor.process(
            Segment::new(&mut stream.queue, handles),
            &mut self.optimizations,
            ExecutionMode::Lazy,
        );
        let size_after = stream.queue.len();

        if size_after != size_before {
            self.free_orphans(handles);
        }

        if size_after == 0 {
            self.streams.remove(&id);
        }
    }

    /// Drain a stream
    pub fn drain(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        if let Some(mut stream) = self.streams.remove(&id) {
            stream.processor.process(
                Segment::new(&mut stream.queue, handles),
                &mut self.optimizations,
                ExecutionMode::Sync,
            );
            self.free_orphans(handles);
        }
    }

    /// When one of the provided streams is different from the current stream, we drain them.
    ///
    /// Returns the current stream id.
    fn resolve_streams(
        &mut self,
        streams: Vec<StreamId>,
        handles: &mut HandleContainer<R::FusionHandle>,
        op: &OperationIr,
    ) -> StreamId {
        let streams = Self::remove_duplicate(streams);
        let current = StreamId::current();

        for id in streams {
            if id != current {
                self.resolve_stream(handles, id, op.nodes());
            }
        }

        current
    }

    /// Drain the stream only if one of the tensor in the given nodes is also included in the
    /// stream queue.
    fn resolve_stream(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        id: StreamId,
        nodes: Vec<&TensorIr>,
    ) {
        if let Some(stream) = self.streams.get(&id) {
            for node in nodes {
                if stream.queue.ids.contains(&node.id) {
                    self.drain(handles, id);
                    return;
                }
            }
        }
    }

    fn remove_duplicate(items: Vec<StreamId>) -> Vec<StreamId> {
        if items.len() == 1 {
            return items;
        }

        let mut output = Vec::with_capacity(items.len());
        for item in items {
            if !output.contains(&item) {
                output.push(item);
            }
        }
        output
    }

    fn free_orphans(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        let nodes = self
            .streams
            .values()
            .flat_map(|a| a.queue.global.iter())
            .flat_map(|a| a.nodes())
            .map(|tensor| &tensor.id)
            .collect::<Vec<_>>();

        handles.free_orphans(&nodes);
    }
}

struct Stream<R: FusionRuntime> {
    queue: OperationQueue<R>,
    processor: Processor<R::Optimization>,
}

#[derive(new)]
struct Segment<'a, R: FusionRuntime> {
    queue: &'a mut OperationQueue<R>,
    handles: &'a mut HandleContainer<R::FusionHandle>,
}

impl<R: FusionRuntime> StreamSegment<R::Optimization> for Segment<'_, R> {
    fn operations(&self) -> &[OperationIr] {
        &self.queue.relative
    }

    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<R::Optimization>) {
        self.queue.execute(id, self.handles, store)
    }
}

impl<R: FusionRuntime> Stream<R> {
    fn new(device: R::FusionDevice) -> Self {
        Self {
            processor: Processor::new(R::optimizations(device)),
            queue: OperationQueue::new(),
        }
    }
}
