use futures_util::{SinkExt, StreamExt};
use std::{collections::HashMap, sync::Arc};
use tokio_tungstenite::{
    connect_async, connect_async_with_config,
    tungstenite::protocol::{Message, WebSocketConfig},
};

use crate::shared::{ConnectionId, Task, TaskResponse, TaskResponseContent};

use super::{router::WsDevice, WsClient};

pub type CallbackSender = tokio::sync::mpsc::Sender<TaskResponseContent>;
pub type CallbackReceiver = tokio::sync::mpsc::Receiver<TaskResponseContent>;

pub enum ClientRequest {
    WithSyncCallback(Task, CallbackSender),
    WithoutCallback(Task),
}

#[derive(Default)]
pub(crate) struct ClientRunner {
    requests: HashMap<ConnectionId, CallbackSender>,
}

impl ClientRunner {
    async fn on_response(&mut self, response: TaskResponse) {
        match self.requests.remove(&response.id) {
            Some(request) => {
                request.send(response.content).await.unwrap();
            }
            None => {
                panic!("Can't ignore message from the server.");
            }
        }
    }

    fn register_callback(&mut self, id: ConnectionId, callback: CallbackSender) {
        self.requests.insert(id, callback);
    }
}

impl ClientRunner {
    pub fn start(device: WsDevice) -> WsClient {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap(),
        );

        let (sender, mut rec) = tokio::sync::mpsc::channel(10);

        let address = format!("{}/{}", device.address.clone(), "ws");
        println!("Starting {address}");

        runtime.spawn(async move {
            println!("Connecting to {address}");
            let (ws_stream, _) = connect_async_with_config(
                address,
                Some(WebSocketConfig {
                    max_send_queue: Some(100),
                    write_buffer_size: 0,
                    max_write_buffer_size: usize::MAX,
                    max_message_size: Some(usize::MAX),
                    max_frame_size: Some(usize::MAX),
                    accept_unmasked_frames: true,
                }),
                true,
            )
            .await
            .expect("Failed to connect");
            let (mut write, mut read) = ws_stream.split();
            let state = Arc::new(tokio::sync::Mutex::new(ClientRunner::default()));

            // Websocket async runner.
            let state_ws = state.clone();
            tokio::spawn(async move {
                let mut start = std::time::Instant::now();
                while let Some(msg) = read.next().await {
                    let msg = match msg {
                        Ok(msg) => msg,
                        Err(err) => panic!("An error happended {err:?}"),
                    };

                    match msg {
                        Message::Binary(bytes) => {
                            let response: TaskResponse = rmp_serde::from_slice(&bytes).unwrap();
                            let mut state = state_ws.lock().await;
                            state.on_response(response).await;
                        }
                        Message::Close(_) => return,
                        _ => panic!("Unsupproted {msg:?}"),
                    };
                    start = std::time::Instant::now();
                }
            });

            // Channel async runner.
            tokio::spawn(async move {
                while let Some(req) = rec.recv().await {
                    let task = match req {
                        ClientRequest::WithSyncCallback(task, callback) => {
                            let mut state = state.lock().await;
                            state.register_callback(task.id, callback);
                            task
                        }
                        ClientRequest::WithoutCallback(task) => task,
                    };
                    let bytes = rmp_serde::to_vec(&task).unwrap();
                    write.send(Message::Binary(bytes)).await.unwrap();
                }
            });
        });

        WsClient::new(device, sender, runtime)
    }
}
