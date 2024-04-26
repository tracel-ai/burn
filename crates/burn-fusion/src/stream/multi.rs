use burn_tensor::repr::{HandleContainer, OperationDescription};

use super::{
    execution::{ExecutionMode, Operation, Processor, StreamSegment},
    store::{ExecutionPlanId, ExecutionPlanStore},
    OperationQueue, StreamId,
};
use crate::FusionBackend;
use std::collections::HashMap;

/// Keep track of multiple concurrent streams of operations.
pub struct MultiStream<B: FusionBackend> {
    streams: HashMap<StreamId, Stream<B>>,
    optimizations: ExecutionPlanStore<B::Optimization>,
    device: B::Device,
}

impl<B: FusionBackend> MultiStream<B> {
    pub(crate) fn new(device: B::Device) -> Self {
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
        desc: OperationDescription,
        operation: Box<dyn Operation<B>>,
        handles: &mut HandleContainer<B>,
    ) {
        let id = self.maybe_drain(streams, handles);

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

        stream.queue.add(desc, operation);

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

    /// Drain the streams.
    pub fn drain(&mut self, handles: &mut HandleContainer<B>, id: StreamId) {
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
    fn maybe_drain(
        &mut self,
        streams: Vec<StreamId>,
        handles: &mut HandleContainer<B>,
    ) -> StreamId {
        let streams = Self::remove_duplicate(streams);
        let current = StreamId::current();

        if streams.len() == 1 {
            // The only case where we don't need to drain.
            if streams[0] == current {
                return current;
            }
        }

        for id in streams {
            self.drain(handles, id);
        }

        current
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

    fn free_orphans(&self, handles: &mut HandleContainer<B>) {
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

struct Stream<B: FusionBackend> {
    queue: OperationQueue<B>,
    processor: Processor<B::Optimization>,
}

#[derive(new)]
struct Segment<'a, B: FusionBackend> {
    queue: &'a mut OperationQueue<B>,
    handles: &'a mut HandleContainer<B>,
}

impl<'i, B: FusionBackend> StreamSegment<B::Optimization> for Segment<'i, B> {
    fn operations(&self) -> &[OperationDescription] {
        &self.queue.relative
    }

    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<B::Optimization>) {
        self.queue.execute(id, self.handles, store)
    }
}

impl<B: FusionBackend> Stream<B> {
    fn new(device: B::Device) -> Self {
        Self {
            processor: Processor::new(B::optimizations(device)),
            queue: OperationQueue::new(),
        }
    }
}
