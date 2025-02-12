use burn_ir::{BackendIr, OperationIr, TensorId, TensorIr};
use burn_router::{Runner, RunnerClient};
use burn_tensor::TensorData;
use core::marker::PhantomData;
use std::sync::mpsc::{Sender, SyncSender};

use crate::shared::{ConnectionId, TaskResponse, TaskResponseContent};

/// The goal of the processor is to asynchronously process compute tasks on it own thread.
pub struct Processor<B: BackendIr> {
    p: PhantomData<B>,
}

pub type Callback<M> = Sender<M>;

pub enum ProcessorTask {
    RegisterOperation(Box<OperationIr>),
    RegisterTensor(TensorId, TensorData),
    ReadTensor(ConnectionId, TensorIr, Callback<TaskResponse>),
    Sync(ConnectionId, Callback<TaskResponse>),
    RegisterOrphan(TensorId),
    Close,
}

impl<B: BackendIr> Processor<B> {
    pub fn start(runner: Runner<B>) -> SyncSender<ProcessorTask> {
        let (sender, rec) = std::sync::mpsc::sync_channel(1);

        std::thread::spawn(move || {
            for item in rec.iter() {
                match item {
                    ProcessorTask::RegisterOperation(op) => {
                        runner.register(*op);
                    }
                    ProcessorTask::RegisterOrphan(id) => {
                        runner.register_orphan(&id);
                    }
                    ProcessorTask::Sync(id, callback) => {
                        let fut = runner.sync();
                        burn_common::future::block_on(fut);
                        callback
                            .send(TaskResponse {
                                content: TaskResponseContent::SyncBackend,
                                id,
                            })
                            .unwrap();
                    }
                    ProcessorTask::RegisterTensor(id, data) => {
                        runner.register_tensor_data_id(id, data);
                    }
                    ProcessorTask::ReadTensor(id, tensor, callback) => {
                        let fut = runner.read_tensor(tensor);
                        let tensor = burn_common::future::block_on(fut);
                        callback
                            .send(TaskResponse {
                                content: TaskResponseContent::ReadTensor(tensor),
                                id,
                            })
                            .unwrap();
                    }
                    ProcessorTask::Close => {
                        let device = runner.device();
                        let fut = runner.sync();
                        burn_common::future::block_on(fut);
                        core::mem::drop(runner);
                        B::sync(&device);
                        return;
                    }
                }
            }
        });

        sender
    }
}
