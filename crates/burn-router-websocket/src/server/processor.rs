use burn_router::{Runner, RunnerClient};
use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::{OperationDescription, ReprBackend, TensorDescription, TensorId},
    DType, TensorData,
};
use core::marker::PhantomData;
use std::sync::mpsc::Sender;

use crate::shared::{ConnectionId, TaskResponse, TaskResponseContent};

/// The goal of the processor is to asynchonously process compute tasks on it own thread.
pub struct Processor<B: ReprBackend> {
    p: PhantomData<B>,
}

pub type Callback<M> = Sender<M>;

pub enum ProcessorTask {
    RegisterOperation(OperationDescription),
    RegisterTensor(ConnectionId, TensorData, Callback<TaskResponse>),
    RegisterTensorEmpty(ConnectionId, Vec<usize>, DType, Callback<TaskResponse>),
    ReadTensor(ConnectionId, TensorDescription, Callback<TaskResponse>),
    Sync(ConnectionId, Callback<TaskResponse>),
    RegisterOrphan(TensorId),
    Close,
}

impl<B: ReprBackend> Processor<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(runner: Runner<B>) -> Sender<ProcessorTask> {
        let (sender, rec) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            for item in rec.iter() {
                match item {
                    ProcessorTask::RegisterOperation(op) => {
                        runner.register(op);
                    }
                    ProcessorTask::RegisterOrphan(id) => {
                        runner.register_orphan(&id);
                    }
                    ProcessorTask::Sync(id, callback) => {
                        runner.sync();
                        let response = TaskResponse {
                            content: TaskResponseContent::SyncBackend,
                            id,
                        };
                        callback.send(response).unwrap();
                    }
                    ProcessorTask::RegisterTensor(id, data, callback) => {
                        let val = runner.register_tensor_data_desc(data);
                        let response = TaskResponse {
                            content: TaskResponseContent::RegisteredTensorEmpty(val),
                            id,
                        };
                        callback.send(response).unwrap();
                    }
                    ProcessorTask::RegisterTensorEmpty(id, shape, dtype, callback) => {
                        let val = runner.register_empty_tensor_desc(shape, dtype);
                        let response = TaskResponse {
                            content: TaskResponseContent::RegisteredTensor(val),
                            id,
                        };
                        callback.send(response).unwrap();
                    }
                    ProcessorTask::ReadTensor(id, tensor, callback) => {
                        let tensor = burn_common::future::block_on(runner.read_tensor(tensor));
                        let response = TaskResponse {
                            content: TaskResponseContent::ReadTensor(tensor),
                            id,
                        };
                        callback.send(response).unwrap();
                    }
                    ProcessorTask::Close => {
                        return;
                    }
                }
            }
        });

        sender
    }
}
