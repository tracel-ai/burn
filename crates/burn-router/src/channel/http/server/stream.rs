use core::marker::PhantomData;
use std::{
    collections::BTreeMap,
    sync::{mpsc::Sender, Arc},
};

use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{ReprBackend, TensorDescription},
    TensorData,
};
use tokio::sync::Mutex;

use crate::{
    http::{
        CloseConnection, ReadTensor, RegisterOperation, RegisterOrphan, RegisterTensor,
        RegisterTensorEmpty, SyncBackend,
    },
    Runner,
};

use super::processor::{Processor, ProcessorTask};

#[derive(Clone)]
/// A stream makes sure all operations registered are executed in the order they were sent to the
/// server, protentially waiting to reconstruct consistency.
pub struct Stream<B: ReprBackend> {
    sender: Sender<ProcessorTask>,
    queue: Arc<Mutex<Queue>>,
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
            queue: Arc::new(Mutex::new(Queue {
                current_index: 0,
                tasks: BTreeMap::new(),
            })),
            _p: PhantomData,
        }
    }

    pub async fn register_operation(&self, op: RegisterOperation) {
        let mut queue = self.queue.lock().await;
        queue
            .tasks
            .insert(op.position, QueuedTask::RegisterOperation(op));
    }

    pub async fn register_tensor(&self, op: RegisterTensor) -> TensorDescription {
        let index = op.position;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender
            .send(ProcessorTask::RegisterTensor(op, sender))
            .unwrap();
        let body = rec.recv().unwrap();

        let mut queue = self.queue.lock().await;
        queue.tasks.insert(index, QueuedTask::Executed);
        body
    }

    pub async fn register_tensor_empty(&self, op: RegisterTensorEmpty) -> TensorDescription {
        let index = op.position;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender
            .send(ProcessorTask::RegisterTensorEmpty(op, sender))
            .unwrap();
        let body = rec.recv().unwrap();

        let mut queue = self.queue.lock().await;
        queue.tasks.insert(index, QueuedTask::Executed);
        body
    }

    pub async fn register_orphan(&self, op: RegisterOrphan) {
        let mut queue = self.queue.lock().await;
        queue
            .tasks
            .insert(op.position, QueuedTask::RegisterOrphan(op));
    }

    pub async fn read_tensor(&self, op: ReadTensor) -> TensorData {
        self.dequeue(op.position).await;

        let index = op.position;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender
            .send(ProcessorTask::ReadTensor(op, sender))
            .unwrap();
        let val = rec.recv().unwrap();

        {
            let mut queue = self.queue.lock().await;
            queue.tasks.insert(index, QueuedTask::Executed);
        };

        val
    }

    pub async fn sync(&self, op: SyncBackend) {
        self.dequeue(op.position).await;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender.send(ProcessorTask::Sync(sender)).unwrap();
        let _val = rec.recv().unwrap();

        let mut queue = self.queue.lock().await;
        queue.tasks.insert(op.position, QueuedTask::Executed);
    }

    pub async fn close(&self, op: CloseConnection) {
        self.dequeue(op.position).await;
        let (sender, rec) = std::sync::mpsc::channel();
        self.sender.send(ProcessorTask::Sync(sender)).unwrap();
        let _val = rec.recv().unwrap();
    }

    // End exclude.
    pub async fn dequeue(&self, end: u64) {
        if end == 0 {
            return;
        }
        loop {
            let mut queue = self.queue.lock().await;
            let key = queue.current_index;

            if key > end - 1 {
                break;
            }

            let task = queue.tasks.remove(&key);
            match task {
                Some(task) => {
                    queue.current_index += 1;
                    match task {
                        QueuedTask::RegisterOperation(op) => {
                            self.sender
                                .send(ProcessorTask::RegisterOperation(op))
                                .unwrap();
                        }
                        QueuedTask::RegisterOrphan(val) => self
                            .sender
                            .send(ProcessorTask::RegisterOrphan(val))
                            .unwrap(),
                        QueuedTask::Executed => {}
                    };
                }
                None => {}
            }
        }
    }
}

struct Queue {
    current_index: u64,
    tasks: BTreeMap<u64, QueuedTask>,
}
enum QueuedTask {
    RegisterOperation(RegisterOperation),
    RegisterOrphan(RegisterOrphan),
    Executed,
}
