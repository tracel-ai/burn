use super::{
    router::WsDevice,
    runner::{ClientRequest, ClientRunner},
};
use crate::shared::{ConnectionId, Task, TaskContent, TaskResponseContent};
use burn_tensor::repr::TensorId;
use std::{
    future::Future,
    sync::{atomic::AtomicU64, Arc},
};
use tokio::sync::mpsc::Sender;

#[derive(Clone)]
pub struct WsClient {
    pub(crate) device: WsDevice,
    pub(crate) sender: Arc<WsSender>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

impl WsClient {
    pub fn init(device: WsDevice) -> Self {
        ClientRunner::start(device)
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
    pub(crate) fn send(&self, content: TaskContent) -> impl Future<Output = ()> + Send {
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

    pub(crate) fn new_tensor_id(&self) -> TensorId {
        let val = self
            .tensor_id_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        TensorId::new(val)
    }
    pub(crate) fn send_callback(
        &self,
        content: TaskContent,
    ) -> impl Future<Output = TaskResponseContent> + Send {
        let position = self
            .position_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sender = self.sender.clone();
        let (callback_sender, mut callback_recv) = tokio::sync::mpsc::channel(1);

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

            let res = match callback_recv.recv().await {
                Some(val) => val,
                None => panic!(""),
            };

            res
        };

        fut
    }
}
