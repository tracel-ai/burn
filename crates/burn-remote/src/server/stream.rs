use core::marker::PhantomData;
use std::sync::mpsc::{Receiver, SyncSender};

use crate::shared::{ConnectionId, TaskResponse};

use super::processor::{Processor, ProcessorTask};
use burn_ir::{BackendIr, OperationIr, TensorId, TensorIr};
use burn_router::Runner;
use burn_tensor::TensorData;

/// A stream makes sure all operations registered are executed in the order they were sent to the
/// server, protentially waiting to reconstruct consistency.
#[derive(Clone)]
pub struct Stream<B: BackendIr> {
    compute_sender: SyncSender<ProcessorTask>,
    writer_sender: SyncSender<Receiver<TaskResponse>>,
    _p: PhantomData<B>,
}

impl<B: BackendIr> Stream<B> {
    pub fn new(runner: Runner<B>, writer_sender: SyncSender<Receiver<TaskResponse>>) -> Self {
        let sender = Processor::start(runner);

        Self {
            compute_sender: sender,
            writer_sender,
            _p: PhantomData,
        }
    }

    pub fn register_operation(&self, op: Box<OperationIr>) {
        self.compute_sender
            .send(ProcessorTask::RegisterOperation(op))
            .unwrap();
    }

    pub fn register_tensor(&self, tensor_id: TensorId, data: TensorData) {
        self.compute_sender
            .send(ProcessorTask::RegisterTensor(tensor_id, data))
            .unwrap()
    }

    pub fn register_orphan(&self, tensor_id: TensorId) {
        self.compute_sender
            .send(ProcessorTask::RegisterOrphan(tensor_id))
            .unwrap()
    }

    pub fn read_tensor(&self, id: ConnectionId, desc: TensorIr) {
        let (callback_sender, callback_rec) = std::sync::mpsc::channel();

        self.compute_sender
            .send(ProcessorTask::ReadTensor(id, desc, callback_sender))
            .unwrap();

        self.writer_sender.send(callback_rec).unwrap();
    }

    pub fn sync(&self, id: ConnectionId) {
        let (callback_sender, callback_rec) = std::sync::mpsc::channel();

        self.compute_sender
            .send(ProcessorTask::Sync(id, callback_sender))
            .unwrap();

        self.writer_sender.send(callback_rec).unwrap();
    }

    pub fn close(&self) {
        self.compute_sender.send(ProcessorTask::Close).unwrap();
    }
}
