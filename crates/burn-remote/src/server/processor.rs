use burn_communication::{
    Protocol,
    data_service::{TensorDataService, TensorTransferId},
};
use burn_ir::{BackendIr, OperationIr, TensorId, TensorIr};
use burn_router::{Runner, RunnerClient};
use burn_tensor::TensorData;
use core::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

use crate::shared::{ConnectionId, TaskResponse, TaskResponseContent, TensorRemote};

/// The goal of the processor is to asynchronously process compute tasks on it own thread.
pub struct Processor<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    p: PhantomData<B>,
    n: PhantomData<P>,
}

pub type Callback<M> = Sender<M>;

pub enum ProcessorTask {
    RegisterOperation(Box<OperationIr>),
    RegisterTensor(TensorId, TensorData),
    RegisterTensorRemote(TensorRemote, TensorId),
    ExposeTensorRemote {
        tensor: TensorIr,
        transfer_id: TensorTransferId,
        count: u32,
    },
    ReadTensor(ConnectionId, TensorIr, Callback<TaskResponse>),
    Sync(ConnectionId, Callback<TaskResponse>),
    Seed(u64),
    Close,
}

impl<B: BackendIr, P> Processor<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub async fn start(
        runner: Runner<B>,
        data_service: Arc<TensorDataService<B, P>>,
    ) -> Sender<ProcessorTask> {
        // channel for tasks to execute
        let (task_sender, mut task_rec) = tokio::sync::mpsc::channel(1);

        tokio::spawn(async move {
            while let Some(item) = task_rec.recv().await {
                match item {
                    ProcessorTask::RegisterOperation(op) => {
                        runner.register(*op);
                    }
                    ProcessorTask::Sync(id, callback) => {
                        runner.sync();
                        callback
                            .send(TaskResponse {
                                content: TaskResponseContent::SyncBackend,
                                id,
                            })
                            .await
                            .unwrap();
                    }
                    ProcessorTask::RegisterTensor(id, data) => {
                        runner.register_tensor_data_id(id, data);
                    }
                    ProcessorTask::RegisterTensorRemote(remote_tensor, new_id) => {
                        log::info!(
                            "Registering remote tensor...(id: {:?})",
                            remote_tensor.transfer_id
                        );
                        let data = data_service
                            .download_tensor(remote_tensor.address, remote_tensor.transfer_id)
                            .await
                            .expect("Can't download tensor: error"); // TODO all these panics should be server errors
                        runner.register_tensor_data_id(new_id, data);
                    }
                    ProcessorTask::ExposeTensorRemote {
                        tensor,
                        transfer_id,
                        count,
                    } => {
                        log::info!("Exposing tensor: (id: {transfer_id:?})");
                        let data = runner.read_tensor(tensor).await;
                        data_service.expose_data(data, count, transfer_id).await;
                    }
                    ProcessorTask::ReadTensor(id, tensor, callback) => {
                        let tensor = runner.read_tensor(tensor).await;
                        callback
                            .send(TaskResponse {
                                content: TaskResponseContent::ReadTensor(tensor),
                                id,
                            })
                            .await
                            .unwrap();
                    }
                    ProcessorTask::Close => {
                        let device = runner.device();
                        runner.sync();
                        core::mem::drop(runner);
                        B::sync(&device);
                        break;
                    }
                    ProcessorTask::Seed(seed) => runner.seed(seed),
                }
            }
        });

        task_sender
    }
}
