use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr, TensorStatus};

use super::{
    StreamId,
    execution::{ExecutionMode, Operation, Processor, StreamSegment},
    queue::OperationQueue,
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{DropOp, FusionRuntime};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Keep track of multiple concurrent lazy streams of operations.
pub struct MultiStream<R: FusionRuntime> {
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    shared_tensors: HashMap<TensorId, SharedTensor>,
    shared_tensors_manual_drop: HashMap<TensorId, TensorIr>,
    // Tensors where their parent stream is closed, but those tensors are still alive, without a
    // timeline.
    refugees: BTreeSet<TensorId>,
    device: R::FusionDevice,
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            shared_tensors: HashMap::new(),
            shared_tensors_manual_drop: HashMap::new(),
            refugees: BTreeSet::new(),
            device,
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
        let (id, analysis) = self.resolve_streams(&streams, handles, &mut repr);

        if self.handle_drop_op(id, &mut repr) {
            return;
        }

        let (num_executed, queue_empty) =
            self.enqueue_operation(id, repr, &streams, operation, handles);

        if num_executed > 0 {
            self.update_shared_tensors(handles, id);
        }

        self.drop_refugees(analysis.refugees, handles, id);

        if queue_empty {
            if let Some(stream) = self.streams.remove(&id) {
                println!(
                    "[{id:?}] Before close stream register {} - c {}",
                    stream.queue.len(),
                    stream.cursor
                );
                self.on_close_stream(id, stream, true);
            }
        }
    }

    fn handle_drop_op(&mut self, id: StreamId, repr: &mut OperationIr) -> bool {
        if let OperationIr::Drop(tensor_ir) = repr {
            println!("[{id:?}] Drop catch on");
            // println!("[{id:?}] {:?}", self.shared_tensors);
            // println!("[{id:?}] {:?}", self.shared_tensors_manual_drop);
            // println!("[{id:?}] {:?}", self.refugees);
            if !matches!(tensor_ir.status, TensorStatus::ReadWrite) {
                let stream = self.streams.get(&id);
                let mut execute_still = false;

                if let Some(shared) = self.shared_tensors.get_mut(&tensor_ir.id) {
                    if stream.is_none() {
                        shared.drop(id);
                        execute_still = shared.streams.is_empty();
                    }
                } else {
                    execute_still = true;
                }

                if execute_still {
                    println!("[{id:?}] Real drop {:?}", tensor_ir.id);
                    self.shared_tensors.remove(&tensor_ir.id);
                    self.shared_tensors_manual_drop.remove(&tensor_ir.id);
                    self.refugees.remove(&tensor_ir.id);
                    tensor_ir.status = TensorStatus::ReadWrite;
                    return false;
                }

                println!("[{id:?}] Skip drop {:?}", tensor_ir.id);
                return true;
            }
        };

        false
    }

    fn on_close_stream(&mut self, stream_id: StreamId, mut stream: Stream<R>, check: bool) {
        println!("[{stream_id:?}] Close stream {:?}", stream.queue.variables);

        for (tensor_id, (origin_stream_id, _latest_status)) in stream.queue.variables.drain() {
            if origin_stream_id == stream_id {
                println!("[{stream_id:?}] Add new refugees from {tensor_id:?}");
                self.refugees.insert(tensor_id);

                match self.shared_tensors.get_mut(&tensor_id) {
                    Some(st) => {
                        println!(
                            "[{stream_id:?}] Remove {:?} from shared tensor since stream is closed",
                            origin_stream_id
                        );
                        st.streams.remove(&origin_stream_id);
                    }
                    None => {}
                }
            }
        }
        // println!("[{stream_id:?}] Num Stream {}", self.streams.len());
        // println!("[{stream_id:?}] Shared Tensors {:?}", self.shared_tensors);
        // println!(
        //     "[{stream_id:?}] Shared Tensors To Drop {:?}",
        //     self.shared_tensors_manual_drop
        // );
        if check {
            println!("[{stream_id:?}] Refugees {:?}", self.refugees);
            for (_id, st) in self.shared_tensors.iter() {
                assert!(!st.streams.contains_key(&stream_id));
            }
        }
    }

    fn enqueue_operation(
        &mut self,
        id: StreamId,
        repr: OperationIr,
        streams: &OperationStreams,
        operation: Box<dyn Operation<R>>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) -> (usize, bool) {
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
        println!("[{id:?}] Train stream from user.");
        self.drain_inner(handles, id);

        if let Some(stream) = self.streams.remove(&id) {
            println!("[{id:?}] Remove stream after drain manually");
            self.on_close_stream(id, stream, true);
        }
    }

    fn drain_inner(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        if let Some(stream) = self.streams.get_mut(&id) {
            println!("[{id:?}] Drain inner");
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
            println!("[{id:?}] Drain inner before on close");
            self.drop_shared_tensors(cleared, handles, id);
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
    ) -> (StreamId, SharedTensorsAnalysis) {
        // TODO: Assumes the server runs on the same thread as user code, which is true with the
        // mutex channel.
        let current = StreamId::current();
        println!("[{current:?}] Operation {op:?}");
        let nodes = op.nodes();

        let analysis = self.analyse_shared_tensors(&nodes, &streams, current);
        Self::cleanup_refugees(&mut self.refugees, &nodes, current, &self.shared_tensors);
        println!("[{current:?}] Analysis: {analysis:?}");

        self.merge_streams_timelines(handles, &analysis, current, &nodes);
        self.register_shared_tensors_drop(&analysis, op);

        (current, analysis)
    }

    fn cleanup_refugees(
        refugees: &mut BTreeSet<TensorId>,
        nodes: &[&TensorIr],
        current: StreamId,
        shared_tensors: &HashMap<TensorId, SharedTensor>,
    ) {
        for node in nodes.iter() {
            if let burn_ir::TensorStatus::ReadWrite = node.status {
                match shared_tensors.get(&node.id) {
                    Some(st) => {
                        if !st.streams.is_empty() {
                            if st.streams.len() == 1 && st.streams.contains_key(&current) {
                            } else {
                                continue;
                            }
                        }
                    }
                    None => {}
                };
                println!("[{current:?}] Drop refugees {node:?}");

                refugees.remove(&node.id);
            }
        }
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
                    println!("[{id:?}] Resolve stream for node {:?}", node.id);
                    self.drain_inner(handles, id);
                    return;
                }
            }
        }
    }

    fn analyse_shared_tensors(
        &mut self,
        nodes: &[&TensorIr],
        streams: &OperationStreams,
        current: StreamId,
    ) -> SharedTensorsAnalysis {
        let mut analysis = SharedTensorsAnalysis::default();

        #[derive(Debug)]
        enum SingleAnalysis {
            FromCurrentStrean,
            FromExistingStream {
                stream_id: StreamId,
                original_cursor: u64,
            },
            /// From a stream that is created, but no operation was executed yet because of lazy
            /// execution.
            FromNewStream {
                stream_id: StreamId,
            },
            IsRefugee,
        }

        println!("{:?}", self.refugees);
        let analyse = |state: &mut SharedTensor, node: &TensorIr| {
            if self.refugees.contains(&node.id) {
                return SingleAnalysis::IsRefugee;
            }

            let stream_id = match streams.streams.get(&node.id) {
                Some(val) => val,
                None => {
                    // When no stream is found, it means the current stream will create the value.
                    return SingleAnalysis::FromCurrentStrean;
                }
            };

            if stream_id == &current {
                return SingleAnalysis::FromCurrentStrean;
            }

            // Here the node is tagged as newly shared.
            let stream_current = self.streams.get(&current);
            let stream = self.streams.get(stream_id);

            state.register_new_stream(current, stream_current);
            match state.register_new_stream(*stream_id, stream) {
                Some(origin) => SingleAnalysis::FromExistingStream {
                    stream_id: *stream_id,
                    original_cursor: origin,
                },
                None => SingleAnalysis::FromNewStream {
                    stream_id: *stream_id,
                },
            }
        };

        println!("[{current:?}] {:?}", self.shared_tensors);
        for node in nodes.iter() {
            match self.shared_tensors.get_mut(&node.id) {
                Some(state) => {
                    let a = analyse(state, &node);
                    println!("[{current:?}] STATE {:?} => {a:?}", node.id);
                    match a {
                        SingleAnalysis::FromCurrentStrean => {
                            analysis.current.push(node.id);
                        }
                        SingleAnalysis::FromExistingStream {
                            stream_id,
                            original_cursor,
                        } => {
                            analysis
                                .existing
                                .push((node.id, stream_id, original_cursor));
                        }
                        SingleAnalysis::FromNewStream { stream_id } => {
                            analysis.new.push((node.id, stream_id));
                        }
                        SingleAnalysis::IsRefugee => {
                            analysis.refugees.push(node.id);
                        }
                    }
                }
                None => {
                    let mut state = SharedTensor::default();

                    let a = analyse(&mut state, &node);
                    println!("[{current:?}] NOSTATE {:?} => {a:?}", node.id);

                    match a {
                        SingleAnalysis::FromCurrentStrean => {
                            // When the current node isn't yet shared we should not register it as
                            // such.
                        }
                        SingleAnalysis::FromExistingStream {
                            stream_id,
                            original_cursor,
                        } => {
                            analysis
                                .existing
                                .push((node.id, stream_id, original_cursor));
                        }
                        SingleAnalysis::FromNewStream { stream_id } => {
                            analysis.new.push((node.id, stream_id));
                        }
                        SingleAnalysis::IsRefugee => {
                            analysis.refugees.push(node.id);
                        }
                    };

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
        current: StreamId,
        nodes: &[&TensorIr],
    ) {
        // If we only have current tensors that are shared, we're safe to not sync the timelines.
        if analysis.new.is_empty() && analysis.existing.is_empty() {
            return;
        }

        let mut streams_to_sync = OperationStreams::default();
        for (tensor_id, stream_id) in analysis.new.iter() {
            streams_to_sync.insert(*tensor_id, *stream_id);
        }

        for (tensor_id, stream_id, original_cursor) in analysis.existing.iter() {
            if let Some(stream) = self.streams.get(stream_id) {
                // We only have to sync stream when the stream isn't up to data to
                // the original cursor of the current operation.
                //
                // This way we're avoiding to re-sync the streams of previsouvly shared tensors
                // that are already resolved.
                if stream.cursor <= *original_cursor {
                    streams_to_sync.insert(*tensor_id, *stream_id);
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
        analysis: &SharedTensorsAnalysis,
        op: &mut OperationIr,
    ) {
        let mut readonly_tensors = Vec::new();

        for (tensor_id, _stream_id) in analysis.new.iter() {
            readonly_tensors.push(*tensor_id);
        }
        for (tensor_id, _stream_id, _cursor) in analysis.existing.iter() {
            readonly_tensors.push(*tensor_id);
        }
        for tensor_id in analysis.current.iter() {
            readonly_tensors.push(*tensor_id);
        }
        println!("Readonly tensors {readonly_tensors:?}");

        for tensor in op.readonly(&readonly_tensors) {
            self.shared_tensors_manual_drop.insert(tensor.id, tensor);
        }
    }

    fn update_shared_tensors(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        id: StreamId,
    ) {
        if !self.shared_tensors.is_empty() {
            let mut to_drop = Vec::new();

            for (stream_id, stream) in self.streams.iter() {
                for (id, state) in self.shared_tensors.iter_mut() {
                    if state.update(*stream_id, stream) {
                        to_drop.push(*id);
                    }
                }
            }
            self.drop_shared_tensors(to_drop, handles, id);
        }
    }

    fn drop_refugees(
        &mut self,
        tensors: Vec<TensorId>,
        handles: &mut HandleContainer<R::FusionHandle>,
        id: StreamId,
    ) {
        let mut to_drop = Vec::new();
        for id in tensors {
            match self.shared_tensors.remove(&id) {
                Some(st) => {
                    if !st.streams.is_empty() {
                        self.shared_tensors.insert(id, st);
                        continue;
                    }
                }
                None => {
                    to_drop.push(id);
                }
            }
        }
        self.drop_shared_tensors(to_drop, handles, id);
    }

    fn drop_shared_tensors(
        &mut self,
        tensors: Vec<TensorId>,
        handles: &mut HandleContainer<R::FusionHandle>,
        stream_id: StreamId,
    ) {
        println!("[{stream_id:?}] Drop shared tensors {tensors:?}");
        for id in tensors {
            self.shared_tensors.remove(&id);
            self.refugees.remove(&id);

            if let Some(tensor) = self.shared_tensors_manual_drop.remove(&id) {
                let streams = OperationStreams::default();

                println!("[{stream_id:?}] Register new drop {id:?}");
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
            "[{id:?}] Update cursors {:?} <= {:?}",
            entry.cursor_current, stream.cursor,
        );

        // We can only free the shared tensor if the latest cursor is executed.
        if entry.cursor_current <= stream.cursor {
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
        self.streams.insert(tensor.id.clone(), tensor.stream);
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
    new: Vec<(TensorId, StreamId)>,
    existing: Vec<(TensorId, StreamId, u64)>,
    refugees: Vec<TensorId>,
}
