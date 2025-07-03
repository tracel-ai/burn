use std::{collections::HashMap, sync::Arc, time::Duration};

use tokio::{
    net::TcpStream,
    runtime::Runtime,
    sync::{
        Mutex,
        mpsc::{Receiver, Sender},
    },
    task::JoinHandle,
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{self, Message, protocol::WebSocketConfig},
};
use tokio_util::sync::CancellationToken;

use crate::global::{
    client::base::ClientRequest,
    shared::{MessageResponse, RemoteRequest, RemoteResponse, RequestId, SessionId},
};

use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;

/// Worker that handles communication with the server for global collective operations.
pub(crate) struct GlobalClientWorker {
    handle: Option<JoinHandle<()>>,
    cancel_token: CancellationToken,
    request_sender: Sender<ClientRequest>,
}

// Rename
struct GlobalClientWorkerState {
    requests: HashMap<RequestId, Sender<RemoteResponse>>,
}

impl GlobalClientWorkerState {
    fn new() -> Self {
        Self {
            requests: HashMap::new(),
        }
    }
}

impl GlobalClientWorker {
    /// Create a new global client worker and start the tasks.
    pub(crate) fn new(
        runtime: &Runtime,
        cancel_token: CancellationToken,
        server_address: &str,
    ) -> Self {
        let (request_sender, request_recv) = tokio::sync::mpsc::channel::<ClientRequest>(10);

        let state = Arc::new(Mutex::new(GlobalClientWorkerState::new()));

        let server_address = server_address.to_owned();
        let handle = runtime.spawn(Self::start(
            state,
            cancel_token.clone(),
            server_address,
            request_recv,
        ));

        Self {
            handle: Some(handle),
            cancel_token,
            request_sender,
        }
    }

    /// Start the global client tasks
    async fn start(
        state: Arc<Mutex<GlobalClientWorkerState>>,
        cancel_token: CancellationToken,
        server_address: String,
        request_recv: Receiver<ClientRequest>,
    ) {
        // Init the connection.
        let address_request = format!("{}/{}", server_address, "request");
        let address_response = format!("{}/{}", server_address, "response");
        let (request, response) = Self::init_connection(address_request, address_response).await;

        // Websocket async worker loading responses from the server.
        tokio::spawn(Self::response_loader(
            state.clone(),
            response,
            cancel_token.clone(),
        ));

        // Channel async worker sending operations to the server.
        tokio::spawn(Self::request_sender(
            request_recv,
            state,
            request,
            cancel_token.clone(),
        ));
    }

    async fn init_connection(
        address_request: String,
        address_response: String,
    ) -> (
        WebSocketStream<MaybeTlsStream<TcpStream>>,
        WebSocketStream<MaybeTlsStream<TcpStream>>,
    ) {
        let stream_request_fut = Self::connect_with_retry(
            address_request.clone(),
            std::time::Duration::from_secs(1),
            None,
        );
        let stream_response_fut = Self::connect_with_retry(
            address_response.clone(),
            std::time::Duration::from_secs(1),
            None,
        );
        let (mut stream_request, mut stream_response) =
            tokio::join!(stream_request_fut, stream_response_fut,);

        let session_id = SessionId::new();
        let req = crate::global::shared::Message::Init(session_id);
        let bytes: tungstenite::Bytes = rmp_serde::to_vec(&req)
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

        (stream_request, stream_response)
    }

    /// Connect with websocket with retries.
    async fn connect_with_retry(
        address: String,
        retry_pause: Duration,
        retry_max: Option<u32>,
    ) -> WebSocketStream<MaybeTlsStream<TcpStream>> {

        const MB: usize = 1024 * 1024;
        let mut retries = 0;

        loop {
            if let Some(max) = retry_max {
                if retries >= max {
                    panic!("Failed to connect to {address} after {max} retries.");
                }
            }

            // Try to connect to the request address.
            println!("Connecting to {address} ...");
            let result = connect_async_with_config(
                address.clone(),
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
            .await;

            if result.is_ok() {
                return result.unwrap().0;
            }
            println!("Failed to connect to {address}, retrying... Attempt #{retries}");
            tokio::time::sleep(retry_pause).await;
            retries += 1;
        }
    }

    /// Unregister the worker and close the connection.
    pub async fn close_connection(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Un-register from server
            let req = RemoteRequest::Finish;
            let resp = self.request(req).await;
            if resp != RemoteResponse::FinishAck {
                panic!("Requested to finish, did not get FinishAck; got {:?}", resp);
            }

            self.cancel_token.cancel();
            handle.await.unwrap();
        }
    }

    async fn response_loader(
        state: Arc<Mutex<GlobalClientWorkerState>>,
        mut stream_response: WebSocketStream<MaybeTlsStream<TcpStream>>,
        cancel_token: CancellationToken,
    ) {
        while !cancel_token.is_cancelled() {
            if let Some(msg) = stream_response.next().await {
                let msg = match msg {
                    Ok(msg) => msg,
                    Err(err) => {
                        panic!(
                            "An error happened while receiving messages from the websocket: {err:?}"
                        )
                    }
                };

                match msg {
                    Message::Binary(bytes) => {
                        let response: MessageResponse = rmp_serde::from_slice(&bytes)
                            .expect("Can deserialize messages from the websocket.");
                        let state_resp = state.lock().await;
                        let response_callback = state_resp
                            .requests
                            .get(&response.id)
                            .expect("Got a response to an unknown request");
                        response_callback.send(response.content).await.unwrap();
                    }
                    Message::Close(_) => {
                        log::warn!("Peer closed the connection");
                        return;
                    }
                    _ => panic!("Unsupported websocket message: {msg:?}"),
                };
            }
        }
        
        eprintln!("Worker closing connection");
        stream_response
            .send(Message::Close(None))
            .await
            .expect("Can send the close message on the websocket.");
    }

    async fn request_sender(
        mut request_recv: Receiver<ClientRequest>,
        worker: Arc<Mutex<GlobalClientWorkerState>>,
        mut stream_request: WebSocketStream<MaybeTlsStream<TcpStream>>,
        cancel_token: CancellationToken,
    ) {
        while !cancel_token.is_cancelled() {
            let Some(request) = request_recv.recv().await else {
                continue;
            };

            let id = RequestId::new();

            // Register the callback if there is one
            {
                let mut state = worker.lock().await;
                state.requests.insert(id, request.callback);
            }

            let request = crate::global::shared::Message::Request(id, request.request);

            let bytes = rmp_serde::to_vec::<crate::global::shared::Message>(&request)
                .expect("Can serialize tasks to bytes.")
                .into();
            stream_request
                .send(Message::Binary(bytes))
                .await
                .expect("Can send the message on the websocket.");
        }
        
        eprintln!("Worker closing connection");
        stream_request
            .send(Message::Close(None))
            .await
            .expect("Can send the close message on the websocket.");
    }

    pub async fn request(&mut self, req: RemoteRequest) -> RemoteResponse {
        let (callback, mut response_recv) = tokio::sync::mpsc::channel::<RemoteResponse>(10);
        let client_req = ClientRequest::new(req, callback);
        self.request_sender.send(client_req).await.unwrap();

        response_recv.recv().await.unwrap()
    }
}
