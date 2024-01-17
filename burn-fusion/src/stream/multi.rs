use std::collections::HashMap;
use std::thread::ThreadId;

use super::{
    execution::{ExecutionMode, Processor, StreamSegment},
    store::{ExecutionPlanId, ExecutionPlanStore},
    Operation, OperationDescription, OperationQueue, StreamId,
};
use crate::{FusionBackend, HandleContainer};

/// Tensor 1 ranomd => stream 1
/// Tensor 2 ranomd => stream 2
///
/// Tensor 3 = tensor 1 + tensor 2.clone()          => stream 1; Lazy
/// Tensor 4 = tensor 2 + aa                        => stream 2; Sync tensor 2 goneeee.
/// Output = Tensor 3 * Tensor 4                    => stream 1; //

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
    pub(crate) fn register(
        &mut self,
        streams: Vec<StreamId>,
        desc: OperationDescription,
        operation: Box<dyn Operation<B>>,
        handles: &mut HandleContainer<B>,
    ) {
        for node in desc.nodes() {
            if let crate::TensorStatus::ReadWrite = node.status {
                let mut streams_to_remove = Vec::new();
                for (key, stream) in self.streams.iter_mut() {
                    let mut should_sync = false;
                    for operation in stream.queue.global.iter() {
                        if should_sync {
                            break;
                        }
                        for node_stream in operation.nodes() {
                            if node_stream.id == node.id {
                                should_sync = true;
                            }
                        }
                    }
                    stream.processor.process(
                        Segment::new(&mut stream.queue, handles),
                        &mut self.optimizations,
                        ExecutionMode::Sync,
                    );
                    streams_to_remove.push(*key);
                }

                for tmp in streams_to_remove {
                    self.streams.remove(&tmp);
                }
            }
        }
        let id = self.merge(streams, handles);
        println!("Register stream [{}] {:?}", id.value, desc);

        self.index_per_thread_add(id);
        let stream = match self.streams.get_mut(&id.value) {
            Some(stream) => {
                println!("Stream exist with {} operations", stream.queue.len());
                stream
            }
            None => {
                println!("Creating stream {}", id.value);
                let stream = Stream::new(self.device.clone());
                self.streams.insert(id.value, stream);
                self.streams
                    .get_mut(&id.value)
                    .expect("Just added, so should be included in the hashmap.")
            }
        };

        stream.queue.add(desc, operation);
        stream.processor.process(
            Segment::new(&mut stream.queue, handles),
            &mut self.optimizations,
            ExecutionMode::Lazy,
        );

        if stream.queue.is_empty() {
            println!("Stream {} completed", id.value);
            self.streams.remove(&id.value);
            self.index_per_thread_remove(id);
        }
    }

    /// Drain the streams.
    pub fn drain(&mut self, handles: &mut HandleContainer<B>, thread_id: ThreadId) {
        println!("Drain on thread {:?}", thread_id);
        // for (id, stream) in self.streams.iter_mut() {
        //     if !stream.queue.is_empty() {
        //         stream.processor.process(
        //             Segment::new(&mut stream.queue, handles),
        //             &mut self.optimizations,
        //             ExecutionMode::Sync,
        //         );
        //     }
        // }
        if let Some(ids) = self.streams_per_thread.remove(&thread_id) {
            println!("Draining stream {ids:?}");
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

    fn merge(&mut self, streams: Vec<StreamId>, handles: &mut HandleContainer<B>) -> StreamId {
        if streams.len() == 1 {
            return streams[0];
        }

        let streams = Self::remove_duplicate(streams);

        if streams.len() == 1 {
            return streams[0];
        }

        let stream_kept = streams[0];
        let mut to_merge = Vec::new();

        for i in 1..streams.len() {
            let stream = streams[i];

            if let Some(item) = self.streams.remove(&stream.value) {
                println!(
                    "-- Merging stream {} with {}",
                    stream_kept.value, stream.value
                );
                to_merge.push((item, stream.thread_id));
            }
        }

        for (stream, thread_id) in to_merge {
            for (desc, op) in stream
                .queue
                .global
                .into_iter()
                .zip(stream.queue.operations.into_iter())
            {
                self.index_per_thread_add(StreamId {
                    value: stream_kept.value,
                    thread_id,
                });
                self.register(vec![stream_kept], desc, op, handles)
            }
        }

        stream_kept
    }

    fn remove_duplicate(items: Vec<StreamId>) -> Vec<StreamId> {
        let mut output = Vec::with_capacity(items.len());
        for item in items {
            if !output.contains(&item) {
                output.push(item);
            }
        }
        output
    }

    fn index_per_thread_add(&mut self, id: StreamId) {
        if let Some(ids) = self.streams_per_thread.get_mut(&id.thread_id) {
            if !ids.contains(&id.value) {
                ids.push(id.value);
            }
        } else {
            self.streams_per_thread.insert(id.thread_id, vec![id.value]);
        }
    }
    fn index_per_thread_remove(&mut self, id: StreamId) {
        let mut should_remove_entry = false;
        if let Some(ids) = self.streams_per_thread.get_mut(&id.thread_id) {
            println!("Remove id {} from thread {:?}", id.value, id.thread_id);
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
