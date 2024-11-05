use core::marker::PhantomData;
use std::{collections::HashMap, sync::mpsc::Sender};

use crate::shared::{ConnectionId, Task, TaskResponse};

use super::processor::{Processor, ProcessorTask};
use burn_router::Runner;
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{OperationDescription, ReprBackend, TensorDescription, TensorId, TensorStatus},
    TensorData,
};

type StreamId = u64;

pub struct StreamManager<B: ReprBackend> {
    runner: Runner<B>,
    tensors: HashMap<TensorId, StreamId>,
    streams: HashMap<StreamId, Stream<B>>,
}

impl<B: ReprBackend> StreamManager<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(runner: Runner<B>) -> Self {
        Self {
            runner,
            tensors: Default::default(),
            streams: Default::default(),
        }
    }
    pub fn select(&mut self, task: &Task) -> Stream<B> {
        let stream_id = task.id.stream_id;

        let mut flushes = Vec::new();
        for (tensor_id, status) in task.content.tensors() {
            let tensor_stream_id = match self.tensors.get(&tensor_id) {
                Some(val) => *val,
                None => {
                    if status != TensorStatus::ReadWrite {
                        self.tensors.insert(tensor_id, stream_id);
                    }
                    continue;
                }
            };
            if tensor_stream_id != stream_id {
                flushes.push(tensor_stream_id);
            }

            if status == TensorStatus::ReadWrite {
                self.tensors.remove(&tensor_id);
            }
        }

        for stream_id in flushes {
            if let Some(stream) = self.streams.get(&stream_id) {
                stream.flush(task.id);
            }
        }

        match self.streams.get(&stream_id) {
            Some(stream) => stream.clone(),
            None => {
                let stream = Stream::<B>::new(self.runner.clone());
                self.streams.insert(stream_id, stream.clone());
                stream
            }
        }
    }
    pub fn close(&mut self) {
        for (id, stream) in self.streams.drain() {
            println!("Closing stream {id}");
            stream.close();
        }
    }
}

/// A stream makes sure all operations registered are executed in the order they were sent to the
/// server, protentially waiting to reconstruct consistency.
#[derive(Clone)]
pub struct Stream<B: ReprBackend> {
    sender: Sender<ProcessorTask>,
    _p: PhantomData<B>,
}

impl<B: ReprBackend> Stream<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(runner: Runner<B>) -> Self {
        let sender = Processor::new(runner);

        Self {
            sender,
            _p: PhantomData,
        }
    }

    pub fn register_operation(&self, op: OperationDescription) {
        self.sender
            .send(ProcessorTask::RegisterOperation(op))
            .unwrap();
    }

    pub fn register_tensor(&self, tensor_id: TensorId, data: TensorData) {
        self.sender
            .send(ProcessorTask::RegisterTensor(tensor_id, data))
            .unwrap()
    }

    pub fn register_orphan(&self, tensor_id: TensorId) {
        self.sender
            .send(ProcessorTask::RegisterOrphan(tensor_id))
            .unwrap()
    }

    pub fn read_tensor(&self, id: ConnectionId, desc: TensorDescription) -> TaskResponse {
        let (callback_sender, callback_rec) = std::sync::mpsc::channel();

        self.sender
            .send(ProcessorTask::ReadTensor(id, desc, callback_sender))
            .unwrap();

        callback_rec.recv().unwrap()
    }

    pub fn sync(&self, id: ConnectionId) -> TaskResponse {
        let (callback_sender, callback_rec) = std::sync::mpsc::channel();

        self.sender
            .send(ProcessorTask::Sync(id, callback_sender))
            .unwrap();

        callback_rec.recv().unwrap()
    }

    pub fn flush(&self, id: ConnectionId) -> TaskResponse {
        let (callback_sender, callback_rec) = std::sync::mpsc::channel();

        self.sender
            .send(ProcessorTask::Flush(id, callback_sender.clone()))
            .unwrap();

        callback_rec.recv().unwrap()
    }

    pub fn close(&self) {
        self.sender.send(ProcessorTask::Close).unwrap();
    }
}
