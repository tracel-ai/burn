use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr, TensorStatus};
use hashbrown::HashSet;

use super::{
    StreamId,
    execution::{ExecutionMode, Operation, Processor, StreamSegment},
    queue::OperationQueue,
    shared_tensors::SharedTensors,
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{
    DropOp, FusionRuntime,
    stream::shared_tensors::{SharedTensorDropped, SingleAnalysis},
};
use std::collections::HashMap;

/// Keep track of multiple concurrent lazy streams of operations.
pub struct MultiStream<R: FusionRuntime> {
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    shared_tensors: SharedTensors,
    device: R::FusionDevice,
}

#[derive(Debug)]
enum DropInfo {
    SkipSharedTensor,
    ForceSharedTensor(StreamId),
    NormalDrop,
    NotDrop,
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            shared_tensors: SharedTensors::default(),
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
        let id = self.resolve_streams(&streams, handles, &mut repr);

        let (id, sync) = match self.handle_drop_op(id, &mut repr) {
            DropInfo::SkipSharedTensor => return,
            DropInfo::NormalDrop => (id, true),
            DropInfo::NotDrop => (id, false),
            DropInfo::ForceSharedTensor(stream_id) => (stream_id, true),
        };

        let (num_executed, queue_empty) =
            self.enqueue_operation(id, repr, &streams, operation, handles);

        if num_executed > 0 {
            if let Some(stream) = self.streams.get(&id) {
                let cleared = self.shared_tensors.on_executed_ops(id, stream);
                let to_drop = self.shared_tensors.clear_tensors(id, cleared);
                self.drop_shared_tensors(to_drop, handles, id);
            }
        }

        let stream = self.streams.get(&id).unwrap();
        let can_remove = self.shared_tensors.can_remove(&id, stream);

        if can_remove {
            if let Some(stream) = self.streams.remove(&id) {
                self.shared_tensors.on_closed_stream(id);
            }
        } else if sync {
            // Cause a problem where the queue becomes corrupted sometimes.
            //
            // TODO: Fix.
            self.drain(handles, id);
        }
    }

    fn handle_drop_op(&mut self, id: StreamId, repr: &mut OperationIr) -> DropInfo {
        let r = match repr {
            OperationIr::Drop(tensor_ir) => {
                match !matches!(tensor_ir.status, TensorStatus::ReadWrite) {
                    true => {
                        let stream = self.streams.get(&id);
                        let on_drop =
                            self.shared_tensors
                                .on_drop(id, tensor_ir.id, stream.is_none());

                        match on_drop {
                            SharedTensorDropped::ForceDrop => {
                                tensor_ir.status = TensorStatus::ReadWrite;
                                DropInfo::ForceSharedTensor(id)
                            }
                            SharedTensorDropped::Skip => DropInfo::SkipSharedTensor,
                        }
                    }
                    false => DropInfo::NormalDrop,
                }
            }
            _ => DropInfo::NotDrop,
        };

        r
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

    /// Mark a tensor as read.
    pub fn mark_read(&mut self, id: StreamId, ir: &TensorIr) {
        if !matches!(ir.status, TensorStatus::ReadWrite) {
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

        // for (id, s) in self.streams.iter() {
        //     println!("{id} => Executed {} Remaining {}", s.cursor, s.queue.len());
        // }
        // println!("Num streams {}", self.streams.len());
        // println!("{:?}", self.shared_tensors);
    }

    /// Drain a stream
    pub fn drain(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        self.drain_inner(handles, id);

        let stream = self.streams.get(&id).unwrap();
        let can_remove = self.shared_tensors.can_remove(&id, stream);

        if can_remove {
            if let Some(_) = self.streams.remove(&id) {
                self.shared_tensors.on_closed_stream(id);
            }
        }
    }

    fn drain_inner(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        if let Some(stream) = self.streams.get_mut(&id) {
            let num_executed = stream.queue.len();
            stream.processor.process(
                Segment::new(&mut stream.queue, handles),
                &mut self.optimizations,
                ExecutionMode::Sync,
            );
            stream.cursor += num_executed as u64;

            let cleared = self.shared_tensors.on_executed_ops(id, &stream);
            let to_drop = self.shared_tensors.clear_tensors(id, cleared);

            self.drop_shared_tensors(to_drop, handles, id);
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
        let current = streams.current;
        let nodes = op.nodes();

        let analysis = self.analyse_shared_tensors(&nodes, &streams, current);
        self.shared_tensors.on_registering_op(current, &nodes);

        self.merge_streams_timelines(handles, &analysis, current, &nodes);
        self.register_shared_tensors_drop(&analysis, op);

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

    fn analyse_shared_tensors(
        &mut self,
        nodes: &[&TensorIr],
        streams: &OperationStreams,
        current: StreamId,
    ) -> SharedTensorsAnalysis {
        let mut shared_analysis = SharedTensorsAnalysis::default();

        for node in nodes.iter() {
            let analysis = self
                .shared_tensors
                .analyse(current, node, streams, &self.streams);
            match analysis {
                SingleAnalysis::SharedFromCurrentStrean => {
                    shared_analysis.current.push(node.id);
                }
                SingleAnalysis::NotShared => {}
                SingleAnalysis::SharedFromExistingStream {
                    stream_id,
                    original_cursor,
                } => {
                    shared_analysis
                        .existing
                        .push((node.id, stream_id, original_cursor));
                }
                SingleAnalysis::SharedFromNewStream { stream_id } => {
                    shared_analysis.new.push((node.id, stream_id));
                }
            }
        }

        shared_analysis
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

        let mut streams_to_sync = HashSet::new();
        for (tensor_id, stream_id) in analysis.new.iter() {
            streams_to_sync.insert(*stream_id);
        }

        for (tensor_id, stream_id, original_cursor) in analysis.existing.iter() {
            if let Some(stream) = self.streams.get(stream_id) {
                // We only have to sync stream when the stream isn't up to data to
                // the original cursor of the current operation.
                //
                // This way we're avoiding to re-sync the streams of previsouvly shared tensors
                // that are already resolved.
                if stream.cursor <= *original_cursor {
                    streams_to_sync.insert(*stream_id);
                }
            }
        }

        for id in streams_to_sync.drain() {
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

        self.shared_tensors
            .tag_manual_drop(op.readonly(&readonly_tensors));
    }

    fn drop_shared_tensors(
        &mut self,
        tensors: Vec<TensorIr>,
        handles: &mut HandleContainer<R::FusionHandle>,
        stream_id: StreamId,
    ) {
        for tensor in tensors {
            let streams = OperationStreams {
                streams: HashMap::new(),
                current: stream_id,
            };

            let op = Box::new(DropOp { id: tensor.id });
            self.register(streams, OperationIr::Drop(tensor), op, handles);
        }
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

#[derive(Debug)]
/// Manage the streams used for the current [operation](OperationIr).
pub struct OperationStreams {
    pub(crate) streams: HashMap<TensorId, StreamId>,
    pub(crate) current: StreamId,
}

impl Default for OperationStreams {
    fn default() -> Self {
        Self {
            streams: HashMap::new(),
            current: StreamId::current(),
        }
    }
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
}
