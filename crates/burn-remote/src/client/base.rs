use super::worker::{ClientRequest, ClientWorker};
use crate::shared::{ComputeTask, ConnectionId, SessionId, Task, TaskResponseContent};
use async_channel::Sender;
use burn_std::id::StreamId;
use burn_communication::ProtocolClient;
use burn_ir::TensorId;
use std::{
    future::Future,
    sync::{Arc, atomic::AtomicU64},
};

pub use super::RemoteDevice;

#[derive(Clone)]
pub struct RemoteClient {
    pub(crate) device: RemoteDevice,
    pub(crate) sender: Arc<RemoteSender>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

impl RemoteClient {
    pub fn init<C: ProtocolClient>(device: RemoteDevice) -> Self {
        ClientWorker::<C>::start(device)
    }

    pub(crate) fn new(
        device: RemoteDevice,
        sender: Sender<ClientRequest>,
        runtime: Arc<tokio::runtime::Runtime>,
        session_id: SessionId,
    ) -> Self {
        Self {
            device,
            runtime,
            sender: Arc::new(RemoteSender {
                sender,
                position_counter: AtomicU64::new(0),
                tensor_id_counter: AtomicU64::new(0),
                session_id,
            }),
        }
    }
}

pub(crate) struct RemoteSender {
    sender: Sender<ClientRequest>,
    position_counter: AtomicU64,
    tensor_id_counter: AtomicU64,
    session_id: SessionId,
}

impl RemoteSender {
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
    ) -> impl Future<Output = TaskResponseContent> + Send + use<> {
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

    pub(crate) fn close(&mut self) {
        let sender = self.sender.clone();

        let close_task = ClientRequest::WithoutCallback(Task::Close(self.session_id));

        sender.send_blocking(close_task).unwrap();
    }
}

impl Drop for RemoteSender {
    fn drop(&mut self) {
        self.close();
    }
}
