use super::{WsClient, runner::WsDevice};
use crate::shared::{
    ComputeTask, ConnectionId, SessionId, Task, TaskResponse, TaskResponseContent,
};
use async_channel::Receiver;
use futures_util::{SinkExt, StreamExt};
use std::{collections::HashMap, sync::Arc};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{
        self,
        protocol::{Message, WebSocketConfig},
    },
};

use tokio_util::sync::CancellationToken;

pub type CallbackSender = async_channel::Sender<TaskResponseContent>;

pub enum ClientRequest {
    Compute(ComputeTask, ConnectionId),
    ComputeWithCallback(ComputeTask, ConnectionId, CallbackSender),
    Close,
}

pub(crate) struct ClientWorker {
    requests: HashMap<ConnectionId, CallbackSender>,
    address_request: String,
    address_response: String,
    stream_response: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    stream_request: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    session_id: Option<SessionId>,
}

enum StreamKind {
    Request,
    Response,
}

impl ClientWorker {
    fn new(address_request: String, address_response: String) -> Self {
        Self {
            requests: HashMap::new(),
            address_request,
            address_response,
            stream_response: None,
            stream_request: None,
            session_id: None,
        }
    }

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

    pub fn start(device: WsDevice) -> WsClient {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap(),
        );

        let (request_sender, request_recv) = async_channel::bounded(10);
        let address_request = format!("{}/{}", device.address.as_str(), "request");
        let address_response = format!("{}/{}", device.address.as_str(), "response");

        #[allow(deprecated)]
        runtime.spawn(Self::start_workers(
            address_request,
            address_response,
            request_recv,
        ));

        WsClient::new(device, request_sender, runtime)
    }

    async fn connect(&mut self) -> Result<(), tungstenite::Error> {
        const MB: usize = 1024 * 1024;

        let session_id = SessionId::new();

        let address_request = self.address_request.clone();
        log::info!("Connecting to {} ...", address_request.clone());
        let (mut stream_request, _) = connect_async_with_config(
            address_request,
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
        .await?;

        let address_response = self.address_response.clone();
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
        .await?;

        // Init the connection.
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

        self.stream_request = Some(stream_request);
        self.stream_response = Some(stream_response);
        self.session_id = Some(session_id);

        Ok(())
    }

    /// Takes the WebSocket stream if it has been initialized,
    /// otherwise initializes it and takes it.
    async fn take_stream(
        &mut self,
        kind: StreamKind,
        close_token: CancellationToken,
    ) -> Option<WebSocketStream<MaybeTlsStream<TcpStream>>> {
        // If stream is none, try to reconnect
        while match kind {
            StreamKind::Request => &self.stream_request,
            StreamKind::Response => &self.stream_response,
        }
        .is_none()
        {
            if close_token.is_cancelled() {
                // Workers have been closed, exit now
                return None;
            }

            // Try to connect
            match self.connect().await {
                Ok(_) => break,
                Err(err) => {
                    log::warn!("Connection failed: {:?}", err);
                }
            }
        }

        let stream = match kind {
            StreamKind::Request => self.stream_request.take(),
            StreamKind::Response => self.stream_response.take(),
        };

        Some(stream.expect("Connection failed, no response stream created."))
    }

    async fn start_workers(
        address_request: String,
        address_response: String,
        request_recv: Receiver<ClientRequest>,
    ) {
        let worker = ClientWorker::new(address_request, address_response);
        let state = Arc::new(tokio::sync::Mutex::new(worker));
        let close_token = CancellationToken::new();

        let state_ws = state.clone();
        let close_token_ws = close_token.clone();
        tokio::spawn(async move {
            Self::handle_responses(state_ws, close_token_ws).await;
        });

        tokio::spawn(async move {
            Self::handle_requests(state, close_token, request_recv).await;
        });
    }

    async fn handle_responses(
        state_ws: Arc<tokio::sync::Mutex<ClientWorker>>,
        close_token: CancellationToken,
    ) {
        let mut stream_response = {
            let mut state = state_ws.lock().await;
            match state
                .take_stream(StreamKind::Response, close_token.clone())
                .await
            {
                Some(stream) => stream,
                None => {
                    log::warn!("Closing response thread");
                    return;
                }
            }
        };

        while let Some(msg) = stream_response.next().await {
            let msg = match msg {
                Ok(msg) => msg,
                Err(err) => {
                    log::warn!("WebSocket stream error: {:?}", err);
                    let mut state = state_ws.lock().await;
                    stream_response = match state
                        .take_stream(StreamKind::Response, close_token.clone())
                        .await
                    {
                        Some(stream) => stream,
                        None => {
                            log::warn!("Closing response thread");
                            return;
                        }
                    };
                    continue;
                }
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
                    continue;
                }
                _ => panic!("Unsupported websocket message: {msg:?}"),
            };

            if close_token.is_cancelled() {
                return;
            }
        }
    }

    async fn handle_requests(
        state: Arc<tokio::sync::Mutex<ClientWorker>>,
        close_token: CancellationToken,
        request_recv: Receiver<ClientRequest>,
    ) {
        let mut stream_request = {
            let mut state = state.lock().await;
            match state
                .take_stream(StreamKind::Request, close_token.clone())
                .await
            {
                Some(stream) => stream,
                None => {
                    log::warn!("Closing request thread");
                    return;
                }
            }
        };

        while let Ok(req) = request_recv.recv().await {
            let task = match req {
                ClientRequest::ComputeWithCallback(task, id, callback) => {
                    let mut state = state.lock().await;
                    state.register_callback(id, callback);
                    Task::Compute(task, id)
                }
                ClientRequest::Compute(task, id) => Task::Compute(task, id),
                ClientRequest::Close => {
                    let session_id = state.lock().await.session_id;
                    match session_id {
                        Some(session_id) => {
                            close_token.cancel();
                            Task::Close(session_id)
                        }
                        None => {
                            log::warn!("Trying to close the session when none is open");
                            close_token.cancel();
                            return;
                        }
                    }
                }
            };

            let bytes = rmp_serde::to_vec(&task)
                .expect("Can serialize tasks to bytes.")
                .into();

            if let Err(err) = stream_request.send(Message::Binary(bytes)).await {
                log::warn!("Request stream error, reopening connection. {:?}", err);

                let mut state = state.lock().await;
                stream_request = match state
                    .take_stream(StreamKind::Request, close_token.clone())
                    .await
                {
                    Some(stream) => stream,
                    None => {
                        log::warn!("Closing request thread");
                        return;
                    }
                };
                continue;
            }
        }

        log::info!("Client closed, shutting down worker.");
    }
}
