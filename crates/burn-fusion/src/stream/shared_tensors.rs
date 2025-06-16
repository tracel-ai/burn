use burn_common::id::StreamId;
use burn_ir::{TensorId, TensorIr};
use hashbrown::HashMap;

use crate::FusionRuntime;

use super::{OperationStreams, Stream};

#[derive(Default)]
pub struct SharedTensors {
    shared_tensors: HashMap<TensorId, SharedTensor>,
    shared_tensors_manual_drop: HashMap<TensorId, TensorIr>,
}

impl core::fmt::Debug for SharedTensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("\n==== Shared Tensors ====\n")?;

        for sh in self.shared_tensors.iter() {
            f.write_fmt(format_args!("  - Shared {}", sh.0))?;
            for (id, state) in sh.1.streams.iter() {
                f.write_fmt(format_args!(
                    " [{}, cursor={}..{}] ",
                    id, state.cursor_origin, state.cursor_current
                ))?;
            }
            f.write_str("\n")?;
        }
        for sh in self.shared_tensors_manual_drop.iter() {
            f.write_fmt(format_args!("  - Manual Drop {}", sh.0))?;
            f.write_str("\n")?;
        }

        f.write_str("========================\n")
    }
}

#[derive(Default, Debug)]
/// A tensor that is shared between multiple streams.
struct SharedTensor {
    streams: HashMap<StreamId, SharedTensorState>,
}

#[derive(Debug)]
struct SharedTensorState {
    cursor_current: u64,
    cursor_origin: u64,
}

#[derive(Debug)]
pub enum SharedTensorDropped {
    ForceDrop(Vec<StreamId>),
    Skip,
}

#[derive(Debug)]
pub enum SingleAnalysis {
    NotShared,
    SharedFromCurrentStrean,
    SharedFromExistingStream {
        stream_id: StreamId,
        original_cursor: u64,
    },
    /// From a stream that is created, but no operation was executed yet because of lazy
    /// execution.
    SharedFromNewStream {
        stream_id: StreamId,
    },
}

impl SharedTensors {
    pub fn on_drop(
        &mut self,
        stream_id: StreamId,
        tensor_id: TensorId,
        stream_gone: bool,
    ) -> SharedTensorDropped {
        let mut execute_still = false;

        if let Some(shared) = self.shared_tensors.get_mut(&tensor_id) {
            if stream_gone {
                shared.drop(stream_id);
                execute_still = shared.streams.is_empty();
            }
        } else {
            execute_still = true;
        }

        if execute_still {
            let state = self.shared_tensors.remove(&tensor_id);
            self.shared_tensors_manual_drop.remove(&tensor_id);

            return match state {
                Some(val) => {
                    let streams = val.streams.keys().copied().collect();
                    SharedTensorDropped::ForceDrop(streams)
                }
                None => SharedTensorDropped::ForceDrop(Vec::new()),
            };
        }

        SharedTensorDropped::Skip
    }

    pub fn on_executed_ops<R: FusionRuntime>(
        &mut self,
        id: StreamId,
        stream: &mut Stream<R>,
    ) -> Vec<TensorId> {
        let mut cleared = Vec::new();
        for (tensor_id, state) in self.shared_tensors.iter_mut() {
            match state.update(id, stream) {
                SharedTensorUpdate::RemovedFromStream(no_more_stream) => {
                    stream.queue.variables.remove(tensor_id);

                    if no_more_stream {
                        cleared.push(*tensor_id);
                    }
                }
                SharedTensorUpdate::NoMoreStream => {
                    cleared.push(*tensor_id);
                }
                SharedTensorUpdate::NothingToDo => {}
            }
        }
        cleared
    }

    pub fn analyse<R: FusionRuntime>(
        &mut self,
        id: StreamId,
        node: &TensorIr,
        streams_op: &OperationStreams,
        streams: &HashMap<StreamId, Stream<R>>,
    ) -> SingleAnalysis {
        let stream_id = match streams_op.streams.get(&node.id) {
            Some(val) => val,
            None => {
                return match self.shared_tensors.contains_key(&node.id) {
                    true => SingleAnalysis::SharedFromCurrentStrean,
                    false => SingleAnalysis::NotShared,
                };
            }
        };

        if stream_id == &id {
            return match self.shared_tensors.contains_key(&node.id) {
                true => SingleAnalysis::SharedFromCurrentStrean,
                false => SingleAnalysis::NotShared,
            };
        }

        // Here the node is tagged as newly shared.
        let stream_current = streams.get(&id);
        let stream = streams.get(stream_id);

        let state = match self.shared_tensors.get_mut(&node.id) {
            Some(state) => state,
            None => {
                self.shared_tensors.insert(node.id, SharedTensor::default());
                self.shared_tensors.get_mut(&node.id).unwrap()
            }
        };

        state.register_new_stream(id, stream_current);
        match state.register_new_stream(*stream_id, stream) {
            Some(origin) => SingleAnalysis::SharedFromExistingStream {
                stream_id: *stream_id,
                original_cursor: origin,
            },
            None => SingleAnalysis::SharedFromNewStream {
                stream_id: *stream_id,
            },
        }
    }

    pub fn on_registering_op(&mut self, id: StreamId, nodes: &[&TensorIr]) {
        for node in nodes {
            if let burn_ir::TensorStatus::ReadWrite = node.status {
                if let Some(st) = self.shared_tensors.get(&node.id) {
                    if !st.streams.is_empty() {
                        if st.streams.len() == 1 && st.streams.contains_key(&id) {
                        } else {
                            continue;
                        }
                    }
                };
            }
        }
    }

    pub fn tag_manual_drop(&mut self, dropped: Vec<TensorIr>) {
        for tensor in dropped {
            if !self.shared_tensors.contains_key(&tensor.id) {
                panic!("What");
            }

            self.shared_tensors_manual_drop.insert(tensor.id, tensor);
        }
    }

    pub fn on_closed_stream(&mut self, id: StreamId) {
        for (_id, st) in self.shared_tensors.iter() {
            assert!(!st.streams.contains_key(&id));
        }
    }

    pub fn clear_tensors(&mut self, tensors: Vec<TensorId>) -> Vec<TensorIr> {
        let mut to_drop = Vec::new();
        for id in tensors {
            self.shared_tensors.remove(&id);

            if let Some(tensor) = self.shared_tensors_manual_drop.remove(&id) {
                to_drop.push(tensor);
            }
        }

        self.register_manual_drop(to_drop)
    }

    fn register_manual_drop(&mut self, mut tensors: Vec<TensorIr>) -> Vec<TensorIr> {
        if self.shared_tensors_manual_drop.is_empty() {
            return tensors;
        }

        let mut to_drop = Vec::new();
        for id in self.shared_tensors_manual_drop.keys() {
            if !self.shared_tensors.contains_key(id) {
                to_drop.push(*id);
            }
        }

        for id in to_drop {
            let entry = self.shared_tensors_manual_drop.remove(&id).unwrap();
            tensors.push(entry);
        }

        tensors
    }
}

pub enum SharedTensorUpdate {
    RemovedFromStream(bool),
    NoMoreStream,
    NothingToDo,
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
            Some(stream) => stream.cursor + stream.queue.global.len() as u64,
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
    fn update<R: FusionRuntime>(&mut self, id: StreamId, stream: &Stream<R>) -> SharedTensorUpdate {
        let entry = match self.streams.remove(&id) {
            Some(val) => val,
            None => {
                return if self.streams.is_empty() {
                    SharedTensorUpdate::NoMoreStream
                } else {
                    SharedTensorUpdate::NothingToDo
                };
            }
        };

        // We can only free the shared tensor if the latest cursor is executed.
        if entry.cursor_current <= stream.cursor {
            SharedTensorUpdate::RemovedFromStream(self.streams.is_empty())
        } else {
            self.streams.insert(id, entry);
            SharedTensorUpdate::NothingToDo
        }
    }

    fn drop(&mut self, id: StreamId) {
        self.streams.remove(&id);
    }
}
