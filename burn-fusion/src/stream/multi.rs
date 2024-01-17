use std::collections::HashMap;
use std::thread::ThreadId;

use super::{
    execution::{ExecutionMode, Processor, StreamSegment},
    store::{ExecutionPlanId, ExecutionPlanStore},
    Operation, OperationDescription, OperationQueue, StreamId,
};
use crate::{FusionBackend, HandleContainer};

/// Keep track of multiple concurrent streams of operations.
///
/// TODO: Actually support multiple streams.
pub struct MultiStream<B: FusionBackend> {
    streams: HashMap<u64, Stream<B>>,
    streams_per_thread: HashMap<ThreadId, Vec<u64>>,
    optimizations: ExecutionPlanStore<B::Optimization>,
    device: B::FusionDevice,
}

impl<B: FusionBackend> MultiStream<B> {
    pub(crate) fn new(device: B::FusionDevice) -> Self {
        Self {
            streams: HashMap::new(),
            streams_per_thread: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            device,
        }
    }

    /// Register a new tensor operation.
    pub fn register(
        &mut self,
        streams: Vec<StreamId>,
        desc: OperationDescription,
        operation: Box<dyn Operation<B>>,
        handles: &mut HandleContainer<B>,
    ) {
        let id = self.merge(streams, handles);

        let stream = if let Some(stream) = self.streams.get_mut(&id.value) {
            stream
        } else {
            let mut stream = Stream::new(self.device);
            self.streams.insert(id.value, stream);
            self.index_per_thread_add(id);
            self.streams
                .get(&id.value)
                .expect("Just added, so should be included in the hashmap.")
        };

        stream.queue.add(desc, operation);
        stream.processor.process(
            Segment::new(&mut stream.queue, handles),
            &mut self.optimizations,
            ExecutionMode::Lazy,
        );

        let is_empty = stream.queue.is_empty();
        core::mem::drop(stream);

        if is_empty {
            self.streams.remove(&id.value);
            self.index_per_thread_remove(id);
        }
    }

    /// Drain the streams.
    pub fn drain(&mut self, handles: &mut HandleContainer<B>, thread_id: ThreadId) {
        if let Some(ids) = self.streams_per_thread.remove(&thread_id) {
            for id in ids {
                if let Some(mut stream) = self.streams.remove(&id) {
                    stream.processor.process(
                        Segment::new(&mut stream.queue, handles),
                        &mut self.optimizations,
                        ExecutionMode::Sync,
                    );
                }
            }
        }
    }

    fn merge(&mut self, mut streams: Vec<StreamId>, handles: &mut HandleContainer<B>) -> StreamId {
        if streams.len() == 1 {
            return streams[0];
        }

        let mut ids = Vec::with_capacity(streams.len());
        for item in streams {
            if !ids.contains(&item) {
                ids.push(item);
            }
        }
        if ids.len() == 1 {
            return ids[0];
        }

        let stream_kept = ids[0];
        let mut streams = Vec::new();

        for i in 1..ids.len() {
            let id = ids[i];

            if let Some(stream) = self.streams.remove(&id.value) {
                streams.push(stream);
            }
        }

        for stream in streams {
            for (desc, op) in stream
                .queue
                .global
                .into_iter()
                .zip(stream.queue.operations.into_iter())
            {
                self.register(vec![stream_kept], desc, op, handles)
            }
        }

        stream_kept
    }

    fn index_per_thread_add(&mut self, id: StreamId) {
        if let Some(ids) = self.streams_per_thread.get_mut(&id.thread_id) {
            ids.push(id.value);
        } else {
            self.streams_per_thread.insert(id.thread_id, vec![id.value]);
        }
    }
    fn index_per_thread_remove(&mut self, id: StreamId) {
        let mut should_remove_entry = false;
        if let Some(ids) = self.streams_per_thread.get_mut(&id.thread_id) {
            ids.retain(|existing| *existing != id.value);
            if ids.is_empty() {
                should_remove_entry = true;
            }
        }

        if should_remove_entry {
            self.streams_per_thread.remove(&id.thread_id);
        }
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
    fn new(device: B::FusionDevice) -> Self {
        Self {
            processor: Processor::new(B::optimizations(device.into())),
            queue: OperationQueue::new(),
        }
    }
}
