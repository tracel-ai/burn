use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr};

use super::{
    OperationQueue, StreamId,
    execution::{ExecutionMode, Operation, Processor, StreamSegment},
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{DropOp, FusionRuntime};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Keep track of multiple concurrent lazy streams of operations.
pub struct MultiStream<R: FusionRuntime> {
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
    shared_tensors: HashMap<TensorId, SharedTensor>,
    shared_tensors_manual_drop: HashMap<TensorId, (TensorIr, StreamId)>,
    exists: BTreeSet<TensorId>,
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            device,
            shared_tensors: HashMap::new(),
            shared_tensors_manual_drop: HashMap::new(),
            exists: BTreeSet::new(),
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
        println!("Execute {repr:?}");
        let id = self.resolve_streams(&streams, handles, &mut repr);
        println!("Executing on stream {id:?}");

        match &repr {
            OperationIr::Drop(tensor_ir) => {
                println!("Drop");
                match tensor_ir.status {
                    burn_ir::TensorStatus::ReadWrite => {}
                    _ => {
                        println!("Skipping drop {:?}", self.shared_tensors);
                        println!("Skipping drop => {:?}", self.streams.len());
                        let stream = self.streams.get(&id);

                        if let Some(shared) = self.shared_tensors.get_mut(&tensor_ir.id) {
                            if stream.is_none() {
                                shared.drop(id);
                            }
                        }
                        return;
                    }
                }
            }
            _ => {}
        };

        let (num_executed, queue_empty) =
            self.enqueue_operation(id, repr, &streams, operation, handles);

        if num_executed > 0 {
            self.check_shared_tensors(handles);
        }

        if queue_empty {
            println!("Removing stream {}", id);
            if let Some(stream) = self.streams.remove(&id) {
                self.on_close_stream(id, stream);
            }
            self.assert_post_remove_stream(&id);
        }
    }

    fn on_close_stream(&mut self, stream_id: StreamId, mut stream: Stream<R>) {
        println!(
            "1 Removing stream with remaining variables {:?}",
            stream.queue.variables
        );
        for (tensor_id, (origin_stream_id, _latest_status)) in stream.queue.variables.drain() {
            if origin_stream_id == stream_id {
                self.exists.insert(tensor_id);
            }
        }
    }

    fn assert_post_remove_stream(&self, stream_id: &StreamId) {
        // for (_, st) in self.shared_tensors.iter() {
        //     assert!(!st.streams.contains_key(stream_id));
        // }
        // for id in self.shared_tensors_manual_drop.values() {
        //     assert_ne!(id.1, *stream_id);
        // }
        println!("Assert {:?}", self.shared_tensors);
        println!("Assert {:?}", self.shared_tensors_manual_drop);
        println!("Exist {:?}", self.exists);
    }

    fn enqueue_operation(
        &mut self,
        id: StreamId,
        repr: OperationIr,
        streams: &OperationStreams,
        operation: Box<dyn Operation<R>>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) -> (usize, bool) {
        println!("{streams:?} on stream {id:?}");
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

        stream.queue.add(repr, operation, streams, id);

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

        (num_executed, queue_empty)
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
            for (tensor_id, state) in self.shared_tensors.iter_mut() {
                if state.update(id, &stream) {
                    cleared.push(*tensor_id);
                }
            }
            self.drop_shared_tensor(cleared, handles);
            for stream in self.streams.iter() {
                println!("{}:{}", stream.0, stream.1.cursor);
            }
            println!("Num Streams {}", self.streams.len());
            println!("{:?}", self.shared_tensors);
            println!(
                "Removing stream with remaining variables {:?}",
                stream.queue.variables
            );
            self.on_close_stream(id, stream);
            self.assert_post_remove_stream(&id);
        }
    }

    /// When one of the provided streams is different from the current stream, we drain them.
    ///
    /// Returns the selected stream id.
    fn resolve_streams(
        &mut self,
        streams: &OperationStreams,
        handles: &mut HandleContainer<R::FusionHandle>,
        op: &mut OperationIr,
    ) -> StreamId {
        let streams_list = streams.to_vec();
        // TODO: Assumes the server runs on the same thread as user code, which is true with the
        // mutex channel.
        let current = StreamId::current();
        let nodes = op.nodes();

        let cleanup_existing_tensors = |exists: &mut BTreeSet<TensorId>| {
            for node in nodes.iter() {
                if let burn_ir::TensorStatus::ReadWrite = node.status {
                    exists.remove(&node.id);
                }
            }
        };

        if streams_list.len() == 0 {
            cleanup_existing_tensors(&mut self.exists);
            return current;
        }

        let analysis = self.register_shared_tensors(&nodes, &streams, current);

        self.merge_streams_timelines(handles, &analysis, &streams, current, &nodes);
        cleanup_existing_tensors(&mut self.exists);

        self.register_shared_tensors_drop(&streams, analysis, op);

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
                if stream.queue.variables.contains_key(&node.id) {
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
    ) -> SharedTensorsAnalysis {
        let mut analysis = SharedTensorsAnalysis::default();

        let register = |state: &mut SharedTensor,
                        node: &TensorIr,
                        shared_tensors: &mut SharedTensorsAnalysis| {
            if self.exists.contains(&node.id) {
                return false;
            }

            if let Some(stream_id) = streams.streams.get(&node.id) {
                if stream_id != &current {
                    let stream = self.streams.get(&current);
                    state.register_new_stream(current, stream);

                    let stream = self.streams.get(stream_id);

                    match state.register_new_stream(*stream_id, stream) {
                        Some(origin) => {
                            shared_tensors.old.push((node.id, origin));
                        }
                        None => {
                            shared_tensors.new.push(node.id);
                        }
                    }

                    return true;
                }
            }

            false
        };

        for node in nodes.iter() {
            match self.shared_tensors.get_mut(&node.id) {
                Some(state) => {
                    if !register(state, &node, &mut analysis) {
                        // If the tensor is currently shared, but we're on the same stream the
                        // tensor was originally created.
                        analysis.current.push(node.id);
                    }
                }
                None => {
                    let mut state = SharedTensor::default();
                    register(&mut state, &node, &mut analysis);

                    if !state.streams.is_empty() {
                        self.shared_tensors.insert(node.id, state);
                    }
                }
            }
        }

        analysis
    }
    fn merge_streams_timelines(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        analysis: &SharedTensorsAnalysis,
        streams: &OperationStreams,
        current: StreamId,
        nodes: &[&TensorIr],
    ) {
        // If we only have current tensors that are shared, we're safe to not sync the timelines.
        if analysis.new.is_empty() && analysis.old.is_empty() {
            return;
        }

        let mut streams_to_sync = OperationStreams::default();
        for i in analysis.new.iter() {
            let stream_id = streams.streams.get(i).unwrap();
            streams_to_sync.insert(*i, *stream_id);
        }

        for (i, original_cursor) in analysis.old.iter() {
            let stream_id = streams.streams.get(i).unwrap();
            if let Some(stream) = self.streams.get(stream_id) {
                // We only have to sync stream when the stream isn't up to data to
                // the original cursor of the current operation.
                //
                // This way we're avoiding to re-sync the streams of previsouvly shared tensors
                // that are already resolved.
                if stream.cursor <= *original_cursor {
                    streams_to_sync.insert(*i, *stream_id);
                }
            }
        }

        let stream_to_sync = streams_to_sync.to_vec();
        for id in stream_to_sync {
            if id != current {
                self.resolve_stream(handles, id, &nodes);
            }
        }
    }

    fn register_shared_tensors_drop(
        &mut self,
        streams: &OperationStreams,
        mut analysis: SharedTensorsAnalysis,
        op: &mut OperationIr,
    ) {
        if analysis.new.is_empty() && analysis.current.is_empty() && analysis.old.is_empty() {
            return;
        }

        let mut readonly_tensors = analysis.new;

        readonly_tensors.append(&mut analysis.current);
        for (i, _) in analysis.old.drain(..) {
            readonly_tensors.push(i);
        }
        // println!("Readonly tensors {readonly_tensors:?}");

        for tensor in op.readonly(&readonly_tensors) {
            println!("Register tensor as drop {tensor:?}");
            println!("{:?}", self.shared_tensors);
            let stream_id = streams.streams.get(&tensor.id).unwrap();
            self.shared_tensors_manual_drop
                .insert(tensor.id, (tensor, *stream_id));
        }
    }

    fn check_shared_tensors(&mut self, handles: &mut HandleContainer<R::FusionHandle>) {
        if !self.shared_tensors.is_empty() {
            // println!("check_shared_tensors {:?}", self.shared_tensors);
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

            if let Some((tensor, stream_id)) = self.shared_tensors_manual_drop.remove(&id) {
                let mut streams = OperationStreams::default();
                // streams.insert(tensor.id, stream_id);

                self.register(
                    streams,
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
    fn close(&mut self) {}
}

#[derive(Default, Debug)]
/// A tensor that is shared between multiple streams.
struct SharedTensor {
    streams: BTreeMap<StreamId, SharedTensorState>,
}

#[derive(Debug)]
struct SharedTensorState {
    cursor_current: u64,
    cursor_origin: u64,
}

impl SharedTensor {
    /// Register the tensor as also part of the given stream.
    ///
    /// The stream might not exist yet when the current tensor is part of the first operation in
    /// the newly created stream.
    fn register_new_stream<R: FusionRuntime>(
        &mut self,
        id: StreamId,
        stream: Option<&Stream<R>>,
    ) -> Option<u64> {
        println!("Register new stream {id:?}");
        let cursor_current = match stream {
            Some(stream) => stream.cursor + stream.queue.len() as u64 + 1,
            None => 1,
        };

        match self.streams.get_mut(&id) {
            Some(s) => {
                s.cursor_current = cursor_current;
                Some(s.cursor_origin)
            }
            None => {
                let state = SharedTensorState {
                    cursor_current,
                    cursor_origin: cursor_current,
                };
                self.streams.insert(id, state);
                None
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

        println!(
            "Update stream={id} entry={entry:?} current={}",
            stream.cursor
        );
        // We can only free the shared tensor if the latest cursor is executed.
        if entry.cursor_current <= stream.cursor {
            println!("Pop stream {}", self.streams.len());
            println!("streams {:?}", self.streams);
            self.streams.is_empty()
        } else {
            self.streams.insert(id, entry);
            false
        }
    }

    fn drop(&mut self, id: StreamId) {
        self.streams.remove(&id);
    }
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

    fn insert(&mut self, id: TensorId, stream_id: StreamId) {
        self.streams.insert(id, stream_id);
    }

    pub(crate) fn get(&self, id: TensorId) -> Option<StreamId> {
        self.streams.get(&id).cloned()
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

#[derive(Default, Debug)]
struct SharedTensorsAnalysis {
    /// Tensors that are shared with other streams, but we're currently executing on the same stream
    /// the tensor was originally created.
    current: Vec<TensorId>,
    /// Tensors that are newly shared with other streams.
    new: Vec<TensorId>,
    old: Vec<(TensorId, u64)>,
}
