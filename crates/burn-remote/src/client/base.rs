use super::worker::{ClientRequest, ClientWorker};
use crate::shared::{ComputeTask, ConnectionId, Task, TaskResponseContent};
use async_channel::Sender;
use burn_common::id::StreamId;
use burn_ir::TensorId;
use std::{
    future::Future,
    sync::{atomic::AtomicU64, Arc},
};

pub use super::WsDevice;

#[derive(Clone)]
pub struct WsClient {
    pub(crate) device: WsDevice,
    pub(crate) sender: Arc<WsSender>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

impl WsClient {
    pub fn init(device: WsDevice) -> Self {
        ClientWorker::start(device)
    }

    pub(crate) fn new(
        device: WsDevice,
        sender: Sender<ClientRequest>,
        runtime: Arc<tokio::runtime::Runtime>,
    ) -> Self {
        Self {
            device,
            runtime,
            sender: Arc::new(WsSender {
                sender,
                position_counter: AtomicU64::new(0),
                tensor_id_counter: AtomicU64::new(0),
            }),
        }
    }
}

pub(crate) struct WsSender {
    sender: Sender<ClientRequest>,
    position_counter: AtomicU64,
    tensor_id_counter: AtomicU64,
}

impl WsSender {
    pub(crate) fn send(&self, task: ComputeTask) {
        let position = self
            .position_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stream_id = StreamId::current();
        let sender = self.sender.clone();

        sender
            .send_blocking(ClientRequest::WithoutCallback(Task::Compute(
                task,
                ConnectionId::new(position, stream_id),
            )))
            .unwrap();
    }

    pub(crate) fn new_tensor_id(&self) -> TensorId {
        let val = self
            .tensor_id_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        TensorId::new(val)
    }
    pub(crate) fn send_callback(
        &self,
        task: ComputeTask,
    ) -> impl Future<Output = TaskResponseContent> + Send {
        let position = self
            .position_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stream_id = StreamId::current();
        let sender = self.sender.clone();
        let (callback_sender, callback_recv) = async_channel::bounded(1);
        sender
            .send_blocking(ClientRequest::WithSyncCallback(
                Task::Compute(task, ConnectionId::new(position, stream_id)),
                callback_sender,
            ))
            .unwrap();

        async move {
            match callback_recv.recv().await {
                Ok(val) => val,
                Err(err) => panic!("{err:?}"),
            }
        }
    }
}
