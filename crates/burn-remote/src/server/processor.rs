use burn_ir::{BackendIr, OperationIr, TensorId, TensorIr};
use burn_router::{Runner, RunnerClient};
use burn_tensor::TensorData;
use core::marker::PhantomData;
use futures_util::{SinkExt, StreamExt};
use std::sync::{
    Arc,
    mpsc::{Sender, SyncSender},
};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{
        self,
        protocol::{Message, WebSocketConfig},
    },
};

use crate::shared::{
    ConnectionId, RemoteTensorReq, TaskResponse, TaskResponseContent, TensorRemote,
};

use super::tensor_data_service::{TensorDataService, TensorExposeState};

/// The goal of the processor is to asynchronously process compute tasks on it own thread.
pub struct Processor<B: BackendIr> {
    p: PhantomData<B>,
}

pub type Callback<M> = Sender<M>;

pub enum ProcessorTask {
    RegisterOperation(Box<OperationIr>),
    RegisterTensor(TensorId, TensorData),
    RegisterTensorRemote(TensorRemote, TensorId),
    ExposeTensorRemote { tensor: TensorIr, count: u32 },
    ReadTensor(ConnectionId, TensorIr, Callback<TaskResponse>),
    Sync(ConnectionId, Callback<TaskResponse>),
    Close,
}

impl<B: BackendIr> Processor<B> {
    pub fn start(runner: Runner<B>, state: Arc<TensorDataService>) -> SyncSender<ProcessorTask> {
        // channel for tasks to execute
        let (task_sender, task_rec) = std::sync::mpsc::sync_channel(1);

        std::thread::spawn(move || {
            for item in task_rec.iter() {
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
                            .unwrap();
                    }
                    ProcessorTask::RegisterTensor(id, data) => {
                        runner.register_tensor_data_id(id, data);
                    }
                    ProcessorTask::RegisterTensorRemote(remote_tensor, new_id) => {
                        // downloading with a websocket requires a Tokio 1.x runtime
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        let data = rt
                            .block_on(Self::download_tensor(remote_tensor))
                            .expect("Could not fetch remote tensor");
                        log::info!("Registering remote tensor...(id: {new_id:?})");
                        runner.register_tensor_data_id(new_id, data);
                    }
                    ProcessorTask::ExposeTensorRemote { tensor, count } => {
                        let id = tensor.id;
                        let fut = runner.read_tensor(tensor);
                        let data = burn_common::future::block_on(fut);
                        let bytes: bytes::Bytes = rmp_serde::to_vec(&data).unwrap().into();

                        let mut exposed_tensors = state.exposed_tensors.lock().unwrap();
                        exposed_tensors.insert(
                            id,
                            TensorExposeState {
                                bytes,
                                max_downloads: count,
                                cur_download_count: 0,
                            },
                        );
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
                        runner.sync();
                        core::mem::drop(runner);
                        B::sync(&device);
                        return;
                    }
                }
            }
        });

        task_sender
    }

    /// Downloads a tensor that is exposed on another server. Requires a Tokio 1.x runtime
    async fn download_tensor(remote_tensor: TensorRemote) -> Option<TensorData> {
        let address_request = format!("{}/{}", remote_tensor.address.as_str(), "data");
        const MB: usize = 1024 * 1024;

        let (mut stream, _) = connect_async_with_config(
            address_request.clone(),
            Some(
                WebSocketConfig::default()
                    .write_buffer_size(0)
                    .max_message_size(None)
                    .max_frame_size(Some(MB * 512))
                    .accept_unmasked_frames(true)
                    .read_buffer_size(64 * 1024), // 64 KiB (previous default)
            ),
            true,
        )
        .await
        .expect("Failed to connect");

        let tensor_request = RemoteTensorReq {
            id: remote_tensor.id,
        };
        let bytes: tungstenite::Bytes = rmp_serde::to_vec(&tensor_request)
            .expect("Can serialize download request to bytes.")
            .into();
        stream
            .send(Message::Binary(bytes.clone()))
            .await
            .expect("Can send the message on the websocket.");

        let id = remote_tensor.id;
        log::info!("Requested tensor (id: {id:?}");

        if let Some(msg) = stream.next().await {
            let msg = match msg {
                Ok(msg) => msg,
                Err(err) => {
                    panic!("An error happened while receiving messages from the websocket: {err:?}")
                }
            };

            match msg {
                Message::Binary(bytes) => {
                    let data: TensorData = rmp_serde::from_slice(&bytes)
                        .expect("Can deserialize messages from the websocket.");
                    return Some(data);
                }
                Message::Close(_) => {
                    log::warn!("Closed connection");
                    return None;
                }
                _ => panic!("Unsupported websocket message: {msg:?}"),
            };
        }

        None
    }
}
