pub use super::RemoteDevice;
use super::worker::{ClientRequest, ClientWorker};
use crate::shared::{ComputeTask, ConnectionId, SessionId, Task, TaskResponseContent};
use async_channel::{RecvError, SendError, Sender};
use burn_communication::ProtocolClient;
use burn_ir::TensorId;
use burn_std::id::StreamId;
use std::{
    future::Future,
    sync::{Arc, atomic::AtomicU64},
};

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

#[allow(unused)]
#[derive(Debug)]
pub enum RemoteSendError {
    SendError(SendError<ClientRequest>),
    RecvError(RecvError),
}

impl RemoteSender {
    /// Generate a new unique (for this [`RemoteSender`] [`TensorId`].
    pub(crate) fn new_tensor_id(&self) -> TensorId {
        TensorId::new(
            self.tensor_id_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Give the next operation sequence number.
    fn next_position(&self) -> u64 {
        self.position_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn send(&self, task: ComputeTask) {
        self.sender
            .send_blocking(ClientRequest::WithoutCallback(Task::Compute(
                task,
                ConnectionId::new(self.next_position(), StreamId::current()),
            )))
            .unwrap();
    }

    pub(crate) fn send_async(
        &self,
        task: ComputeTask,
    ) -> impl Future<Output = Result<TaskResponseContent, RemoteSendError>> + Send + use<> {
        let stream_id = StreamId::current();
        let position = self.next_position();
        let sender = self.sender.clone();

        async move {
            let (tx, rx) = async_channel::bounded(1);

            if let Err(e) = sender
                .send(ClientRequest::WithSyncCallback(
                    Task::Compute(task, ConnectionId::new(position, stream_id)),
                    tx,
                ))
                .await
            {
                return Err(RemoteSendError::SendError(e));
            }

            match rx.recv().await {
                Ok(response) => Ok(response),
                Err(e) => Err(RemoteSendError::RecvError(e)),
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
