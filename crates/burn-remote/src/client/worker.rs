use super::{WsClient, runner::WsDevice};
use crate::shared::{ConnectionId, SessionId, Task, TaskResponse, TaskResponseContent};
use futures_util::{SinkExt, StreamExt};
use std::{collections::HashMap, sync::Arc};
use tokio_tungstenite::{
    connect_async_with_config, tungstenite,
    tungstenite::protocol::{Message, WebSocketConfig},
};

pub type CallbackSender = async_channel::Sender<TaskResponseContent>;

pub enum ClientRequest {
    WithSyncCallback(Task, CallbackSender),
    WithoutCallback(Task),
}

#[derive(Default)]
pub(crate) struct ClientWorker {
    requests: HashMap<ConnectionId, CallbackSender>,
}

impl ClientWorker {
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

impl ClientWorker {
    pub fn start(device: WsDevice) -> WsClient {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap(),
        );

        let (sender, rec) = async_channel::bounded(10);
        let address_request = format!("{}/{}", device.address.as_str(), "request");
        let address_response = format!("{}/{}", device.address.as_str(), "response");

        const MB: usize = 1024 * 1024;

        #[allow(deprecated)]
        runtime.spawn(async move {
            log::info!("Connecting to {address_request} ...");
            let (mut stream_request, _) = connect_async_with_config(
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
            let (mut stream_response, _) = connect_async_with_config(
                address_response,
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

            let state = Arc::new(tokio::sync::Mutex::new(ClientWorker::default()));

            // Init the connection.
            let session_id = SessionId::new();
            let bytes: tungstenite::Bytes = rmp_serde::to_vec(&Task::Init(session_id))
                .expect("Can serialize tasks to bytes.")
                .into();
            stream_request
                .send(Message::Binary(bytes.clone()))
                .await
                .expect("Can send the message on the websocket.");
            stream_response
                .send(Message::Binary(bytes))
                .await
                .expect("Can send the message on the websocket.");

            // Websocket async worker loading callback from the server.
            let state_ws = state.clone();
            tokio::spawn(async move {
                while let Some(msg) = stream_response.next().await {
                    let msg = match msg {
                        Ok(msg) => msg,
                        Err(err) => panic!(
                            "An error happened while receiving messages from the websocket: {err:?}"
                        ),
                    };

                    match msg {
                        Message::Binary(bytes) => {
                            let response: TaskResponse = rmp_serde::from_slice(&bytes)
                                .expect("Can deserialize messages from the websocket.");
                            let mut state = state_ws.lock().await;
                            state.on_response(response).await;
                        }
                        Message::Close(_) => {
                            log::warn!("Closed connection");
                            return;
                        }
                        _ => panic!("Unsupported websocket message: {msg:?}"),
                    };
                }
            });

            // Channel async worker sending operations to the server.
            tokio::spawn(async move {
                while let Ok(req) = rec.recv().await {
                    let task = match req {
                        ClientRequest::WithSyncCallback(task, callback) => {
                            if let Task::Compute(_content, id) = &task {
                                let mut state = state.lock().await;
                                state.register_callback(*id, callback);
                            }
                            task
                        }
                        ClientRequest::WithoutCallback(task) => task,
                    };
                    let bytes = rmp_serde::to_vec(&task)
                        .expect("Can serialize tasks to bytes.")
                        .into();
                    stream_request
                        .send(Message::Binary(bytes))
                        .await
                        .expect("Can send the message on the websocket.");
                }
            });
        });

        WsClient::new(device, sender, runtime)
    }
}
