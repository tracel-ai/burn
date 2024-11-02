use burn_tensor::{
    repr::{OperationDescription, TensorDescription},
    TensorData,
};
use std::{
    future::Future,
    sync::{atomic::AtomicU64, Arc},
};
use tokio::sync::mpsc::Sender;

use super::runner::{CallbackReceiver, ClientRequest, ClientRunner};
use crate::shared::{ConnectionId, Task, TaskContent, TaskResponseContent};

#[derive(Clone)]
pub struct WebSocketClient {
    sender: Arc<WsSender>,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl WebSocketClient {
    pub fn init(address: &str) -> Self {
        ClientRunner::start(address.to_string())
    }

    pub(crate) fn new(
        sender: Sender<ClientRequest>,
        runtime: Arc<tokio::runtime::Runtime>,
    ) -> Self {
        Self {
            runtime,
            sender: Arc::new(WsSender {
                sender,
                position_counter: AtomicU64::new(0),
            }),
        }
    }

    pub fn register(&self, op: OperationDescription) {
        log::info!("Register Operation.");
        let fut = self.sender.send(TaskContent::RegisterOperation(op));
        self.runtime.spawn(fut);
    }

    pub fn read_tensor(&self, desc: TensorDescription) -> TensorData {
        let (fut, callback) = self.sender.send_callback(TaskContent::ReadTensor(desc));

        self.runtime.block_on(fut);

        match callback.recv() {
            Ok(msg) => match msg {
                TaskResponseContent::ReadTensor(data) => data,
                _ => panic!("Invalid message type {msg:?}"),
            },
            Err(err) => panic!("Unable to read tensor {err:?}"),
        }
    }
}

struct WsSender {
    sender: Sender<ClientRequest>,
    position_counter: AtomicU64,
}

impl WsSender {
    fn send(&self, content: TaskContent) -> impl Future<Output = ()> + Send {
        let position = self
            .position_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sender = self.sender.clone();

        async move {
            sender
                .send(ClientRequest::WithoutCallback(Task {
                    content,
                    id: ConnectionId::new(position),
                }))
                .await
                .unwrap();
        }
    }

    fn send_callback(
        &self,
        content: TaskContent,
    ) -> (impl Future<Output = ()> + Send, CallbackReceiver) {
        let position = self
            .position_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sender = self.sender.clone();
        let (callback_sender, callback_recv) = std::sync::mpsc::channel();

        let fut = async move {
            sender
                .send(ClientRequest::WithSyncCallback(
                    Task {
                        content,
                        id: ConnectionId::new(position),
                    },
                    callback_sender,
                ))
                .await
                .unwrap();
        };

        (fut, callback_recv)
    }
}
