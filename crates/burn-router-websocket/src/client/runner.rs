use futures_util::{SinkExt, StreamExt};
use std::{
    collections::HashMap,
    sync::{mpsc::Sender, Arc},
};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use crate::shared::{ConnectionId, Task, TaskResponse, TaskResponseContent};

use super::WebSocketClient;

pub type CallbackSender = std::sync::mpsc::Sender<TaskResponseContent>;
pub type CallbackReceiver = std::sync::mpsc::Receiver<TaskResponseContent>;

pub enum ClientRequest {
    WithSyncCallback(Task, CallbackSender),
    WithoutCallback(Task),
}

#[derive(Default)]
pub(crate) struct ClientRunner {
    requests: HashMap<ConnectionId, CallbackSender>,
}

impl ClientRunner {
    fn on_response(&mut self, response: TaskResponse) {
        match self.requests.remove(&response.id) {
            Some(request) => {
                request.send(response.content).unwrap();
            }
            None => {
                panic!("Can't ignore message from the server.");
            }
        }
    }

    fn register_callback(&mut self, id: ConnectionId, callback: Sender<TaskResponseContent>) {
        self.requests.insert(id, callback);
    }
}

impl ClientRunner {
    pub fn start(address: String) -> WebSocketClient {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );

        let (sender, mut rec) = tokio::sync::mpsc::channel(100);

        runtime.spawn(async move {
            let (ws_stream, _) = connect_async(address).await.expect("Failed to connect");
            let (mut write, read) = ws_stream.split();
            let state = Arc::new(tokio::sync::Mutex::new(ClientRunner::default()));

            // Websocket async runner.
            let state_ws = state.clone();
            tokio::spawn(async move {
                read.for_each(|msg| async {
                    let msg = match msg {
                        Ok(msg) => msg,
                        Err(err) => panic!("An error happended {err:?}"),
                    };

                    match msg {
                        Message::Binary(bytes) => {
                            let mut state = state_ws.lock().await;
                            let response: TaskResponse = rmp_serde::from_slice(&bytes).unwrap();
                            state.on_response(response);
                        }
                        Message::Close(_) => return,
                        _ => panic!("Unsupproted {msg:?}"),
                    };
                })
                .await;
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

        WebSocketClient::new(sender, runtime)
    }
}
