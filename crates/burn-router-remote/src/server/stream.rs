use core::marker::PhantomData;
use std::sync::mpsc::{Receiver, Sender};

use crate::shared::{ConnectionId, TaskResponse};

use super::processor::{Processor, ProcessorTask};
use burn_router::Runner;
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{OperationDescription, ReprBackend, TensorDescription, TensorId},
    TensorData,
};

/// A stream makes sure all operations registered are executed in the order they were sent to the
/// server, protentially waiting to reconstruct consistency.
pub struct Stream<B: ReprBackend> {
    sender: Sender<ProcessorTask>,
    callback_sender: Sender<TaskResponse>,
    callback_rec: Receiver<TaskResponse>,
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
        let (callback_sender, callback_rec) = std::sync::mpsc::channel();

        Self {
            sender,
            callback_sender,
            callback_rec,
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
        self.sender
            .send(ProcessorTask::ReadTensor(
                id,
                desc,
                self.callback_sender.clone(),
            ))
            .unwrap();

        self.callback_rec.recv().unwrap()
    }

    pub fn sync(&self, id: ConnectionId) -> TaskResponse {
        self.sender
            .send(ProcessorTask::Sync(id, self.callback_sender.clone()))
            .unwrap();
        self.callback_rec.recv().unwrap()
    }

    pub fn flush(&self, id: ConnectionId) -> TaskResponse {
        self.sender
            .send(ProcessorTask::Flush(id, self.callback_sender.clone()))
            .unwrap();
        self.callback_rec.recv().unwrap()
    }

    pub fn close(&self) {
        self.sender.send(ProcessorTask::Close).unwrap();
    }
}
