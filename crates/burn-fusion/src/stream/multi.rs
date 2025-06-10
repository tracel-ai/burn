use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr};

use super::{
    OperationQueue, StreamId,
    execution::{ExecutionMode, Operation, Processor, StreamSegment},
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{DropOp, FusionRuntime};
use std::collections::{BTreeMap, HashMap};

/// Keep track of multiple concurrent streams of operations.
pub struct MultiStream<R: FusionRuntime> {
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
    shared_tensors: HashMap<TensorId, SharedTensor>,
    shared_tensors_manual_drop: HashMap<TensorId, TensorIr>,
}

#[derive(Default, Debug)]
/// Manage the streams used for the current [operation](OperationIr).
pub struct OperationStreams {
    streams: HashMap<TensorId, StreamId>,
}

impl OperationStreams {
    /// Register a tensor in the list of streams used for the current [operation](OperationIr).
    ///
    /// You only need to register input tensors, not the outputs.
    /// So init tensor operations should have no streams registered.
    pub fn tensor<R: FusionRuntime>(&mut self, tensor: &crate::FusionTensor<R>) {
        self.streams
            .insert(tensor.id.as_ref().clone(), tensor.stream);
    }

    fn to_vec(&self) -> Vec<StreamId> {
        let mut streams = Vec::new();
        for stream in self.streams.values() {
            if !streams.contains(stream) {
                streams.push(*stream);
            }
        }

        streams
    }
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            device,
            shared_tensors: HashMap::new(),
            shared_tensors_manual_drop: HashMap::new(),
        }
    }

    /// Register a new tensor operation.
    pub(crate) fn register(
        &mut self,
        streams: OperationStreams,
        mut repr: OperationIr,
        operation: Box<dyn Operation<R>>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let id = self.resolve_streams(streams, handles, &mut repr);

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

        let len_before = stream.queue.len();
        stream.processor.process(
            Segment::new(&mut stream.queue, handles),
            &mut self.optimizations,
            ExecutionMode::Lazy,
        );
        let len_after = stream.queue.len();
        let num_executed = len_before - len_after;

        stream.cursor += num_executed as u64;
        let queue_empty = stream.queue.is_empty();

        if num_executed > 0 {
            self.check_shared_tensors(handles);
        }

        if queue_empty {
            self.streams.remove(&id);
        }
    }

    /// Drain a stream
    pub fn drain(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        if let Some(mut stream) = self.streams.remove(&id) {
            let num_executed = stream.queue.len();
            stream.processor.process(
                Segment::new(&mut stream.queue, handles),
                &mut self.optimizations,
                ExecutionMode::Sync,
            );
            stream.cursor += num_executed as u64;

            let mut cleared = Vec::new();
            for (tid, state) in self.shared_tensors.iter_mut() {
                if state.update(id, &stream) {
                    cleared.push(*tid);
                }
            }
            self.drop_shared_tensor(cleared, handles);
        }
    }

    /// When one of the provided streams is different from the current stream, we drain them.
    ///
    /// Returns the current stream id.
    fn resolve_streams(
        &mut self,
        streams: OperationStreams,
        handles: &mut HandleContainer<R::FusionHandle>,
        op: &mut OperationIr,
    ) -> StreamId {
        let streams_list = streams.to_vec();
        let current = StreamId::current();

        if streams_list.len() == 0 || (streams_list.len() == 1 && streams_list.contains(&current)) {
            return current;
        }

        let nodes = op.nodes();

        let shared_tensors_op = self.register_shared_tensors(&nodes, &streams, current);

        for id in streams_list {
            if id != current {
                self.resolve_stream(handles, id, &nodes);
            }
        }

        // Important when used by multiple lazy streams.
        for tensor in op.readonly(&shared_tensors_op) {
            self.shared_tensors_manual_drop.insert(tensor.id, tensor);
        }

        current
    }

    /// Drain the stream only if one of the tensor in the given nodes is also included in the
    /// stream queue.
    fn resolve_stream(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        id: StreamId,
        nodes: &[&TensorIr],
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

    fn register_shared_tensors(
        &mut self,
        nodes: &[&TensorIr],
        streams: &OperationStreams,
        current: StreamId,
    ) -> Vec<TensorId> {
        let mut shared_tensors = Vec::new();

        let mut register = |state: &mut SharedTensor, node: &TensorIr| {
            if let Some(stream_id) = streams.streams.get(&node.id) {
                if stream_id != &current {
                    shared_tensors.push(node.id);

                    let stream = self.streams.get(&current);
                    state.register_new_stream(current, stream);

                    let stream = self.streams.get(stream_id);
                    state.register_new_stream(*stream_id, stream);
                }
            }
        };

        for node in nodes.iter() {
            match self.shared_tensors.get_mut(&node.id) {
                Some(state) => register(state, &node),
                None => {
                    let mut state = SharedTensor::default();
                    register(&mut state, &node);

                    if !state.streams.is_empty() {
                        self.shared_tensors.insert(node.id, state);
                    }
                }
            }
        }

        shared_tensors
    }

    fn check_shared_tensors(&mut self, handles: &mut HandleContainer<R::FusionHandle>) {
        if !self.shared_tensors.is_empty() {
            let mut to_drop = Vec::new();

            for (stream_id, stream) in self.streams.iter() {
                for (id, state) in self.shared_tensors.iter_mut() {
                    if state.update(*stream_id, stream) {
                        to_drop.push(*id);
                    }
                }
            }
            self.drop_shared_tensor(to_drop, handles);
        }
    }

    fn drop_shared_tensor(
        &mut self,
        tensors: Vec<TensorId>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        for id in tensors {
            self.shared_tensors.remove(&id);

            if let Some(tensor) = self.shared_tensors_manual_drop.remove(&id) {
                self.register(
                    OperationStreams::default(),
                    OperationIr::Drop(tensor),
                    Box::new(DropOp { id }),
                    handles,
                );
            }
        }
    }
}

struct Stream<R: FusionRuntime> {
    queue: OperationQueue<R>,
    processor: Processor<R::Optimization>,
    cursor: u64,
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
            cursor: 0,
        }
    }
}

#[derive(Default, Debug)]
/// A tensor that is shared between multiple streams.
struct SharedTensor {
    streams: BTreeMap<StreamId, u64>,
}

impl SharedTensor {
    /// Register the tensor as also part of the given stream.
    ///
    /// The stream might not exist yet when the current tensor is part of the first operation in
    /// the newly created stream.
    fn register_new_stream<R: FusionRuntime>(&mut self, id: StreamId, stream: Option<&Stream<R>>) {
        match stream {
            Some(stream) => {
                let cursor = stream.cursor + stream.queue.len() as u64 + 1;

                self.streams.insert(id, cursor);
            }
            None => {
                self.streams.insert(id, 1);
            }
        }
    }

    /// Update the current shared tensor state on the given stream.
    ///
    /// If the shared tensor is no longer needed on the stream, we will remove it from the list of
    /// shared streams.
    ///
    /// If the shared tensor is needed on no stream, we return true, indicating that the shared
    /// tensor is safe to manually drop.
    fn update<R: FusionRuntime>(&mut self, id: StreamId, stream: &Stream<R>) -> bool {
        let entry = match self.streams.remove(&id) {
            Some(val) => val,
            None => return self.streams.is_empty(),
        };

        if entry <= stream.cursor {
            self.streams.is_empty()
        } else {
            self.streams.insert(id, entry);
            false
        }
    }
}
