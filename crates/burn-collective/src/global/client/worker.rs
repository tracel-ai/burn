use std::{collections::HashMap, marker::PhantomData, sync::Arc, time::Duration};

use burn_network::network::{NetworkClient, NetworkStream};
use tokio::{
    runtime::Runtime,
    sync::{
        Mutex,
        mpsc::{Receiver, Sender},
    },
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use crate::global::shared::base::{
    Message, MessageResponse, RemoteRequest, RemoteResponse, RequestId, SessionId,
};

/// Worker that handles communication with the server for global collective operations.
pub(crate) struct GlobalClientWorker<N: NetworkClient> {
    handle: Option<JoinHandle<()>>,
    cancel_token: CancellationToken,
    request_sender: Sender<ClientRequest>,
    _phantom_data: PhantomData<N>,
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

#[derive(Debug)]
pub(crate) struct ClientRequest {
    pub request: RemoteRequest,
    pub callback: Sender<RemoteResponse>,
}

impl ClientRequest {
    pub fn new(request: RemoteRequest, callback: Sender<RemoteResponse>) -> Self {
        Self { request, callback }
    }
}

impl<C: NetworkClient> GlobalClientWorker<C> {
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
            _phantom_data: PhantomData,
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
        let response_handle = tokio::spawn(Self::response_loader(
            state.clone(),
            response,
            cancel_token.clone(),
        ));

        // Channel async worker sending operations to the server.
        let request_handle = tokio::spawn(Self::request_sender(
            request_recv,
            state,
            request,
            cancel_token.clone(),
        ));

        let (Ok(_), Ok(_)) = tokio::join!(response_handle, request_handle,) else {
            panic!("Failed to join global collective client worker tasks.");
        };
    }

    async fn init_connection(
        address_request: String,
        address_response: String,
    ) -> (C::ClientStream, C::ClientStream) {
        let session_id = SessionId::new();

        let stream_request_fut = Self::connect_with_retry(
            address_request.clone(),
            std::time::Duration::from_secs(1),
            None,
            session_id,
        );
        let stream_response_fut = Self::connect_with_retry(
            address_response.clone(),
            std::time::Duration::from_secs(1),
            None,
            session_id,
        );

        tokio::join!(stream_request_fut, stream_response_fut,)
    }

    /// Connect with websocket with retries.
    async fn connect_with_retry(
        address: String,
        retry_pause: Duration,
        retry_max: Option<u32>,
        session_id: SessionId,
    ) -> C::ClientStream {
        let mut retries = 0;
        loop {
            if let Some(max) = retry_max {
                if retries >= max {
                    panic!("Failed to connect to {address} after {max} retries.");
                }
            }

            // Try to connect to the request address.
            println!("Connecting to {address} ...");
            let result = C::connect(address.clone()).await;

            if let Some(mut stream) = result {
                let init_msg = Message::Init(session_id);
                let bytes: bytes::Bytes = rmp_serde::to_vec(&init_msg).unwrap().into();
                stream
                    .send(bytes)
                    .await
                    .expect("Can send the init message on the websocket.");
                return stream;
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
                panic!("Requested to finish, did not get FinishAck; got {resp:?}");
            }

            self.cancel_token.cancel();
            eprintln!("Cancelling worker tasks...");

            handle.await.unwrap();
        }
    }

    async fn response_loader(
        state: Arc<Mutex<GlobalClientWorkerState>>,
        mut stream_response: C::ClientStream,
        cancel_token: CancellationToken,
    ) {
        loop {
            tokio::select! {
                // Check if the cancel token is cancelled
                _ = cancel_token.cancelled() => {
                    break;
                }
                // .. Or get a message from the websocket
                response = stream_response.recv() => {
                    match response {
                        Err(err) => {
                            eprintln!("Error receiving message from websocket: {err:?}");
                            break;
                        }
                        Ok(response) => {
                            let Some(response) = response else {
                                eprintln!("Closed connection");
                                break;
                            };

                            let response: MessageResponse = rmp_serde::from_slice(&response.data)
                                .expect("Can deserialize messages from the websocket.");
                            let state_resp = state.lock().await;
                            let response_callback = state_resp
                                .requests
                                .get(&response.id)
                                .expect("Got a response to an unknown request");
                            response_callback.send(response.content).await.unwrap();
                        }
                    }
                }
            }
        }

        eprintln!("Worker closing connection");
        stream_response
            .close()
            .await
            .expect("Can close the websocket stream.");
    }

    async fn request_sender(
        mut request_recv: Receiver<ClientRequest>,
        worker: Arc<Mutex<GlobalClientWorkerState>>,
        mut stream_request: C::ClientStream,
        cancel_token: CancellationToken,
    ) {
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    break;
                },
                request = request_recv.recv() => {
                    let Some(request) = request else {
                        continue;
                    };

                    let id = RequestId::new();

                    // Register the callback if there is one
                    {
                        let mut state = worker.lock().await;
                        state.requests.insert(id, request.callback);
                    }

                    let request = Message::Request(id, request.request);

                    let bytes = rmp_serde::to_vec::<Message>(&request)
                        .expect("Can serialize tasks to bytes.")
                        .into();
                    stream_request
                        .send(bytes)
                        .await
                        .expect("Can send the message on the websocket.");
                }
            }
        }

        eprintln!("Worker closing connection");
        stream_request
            .close()
            .await
            .expect("Can send the close message on the websocket.");
    }

    pub async fn request(&self, req: RemoteRequest) -> RemoteResponse {
        let (callback, mut response_recv) = tokio::sync::mpsc::channel::<RemoteResponse>(10);
        let client_req = ClientRequest::new(req, callback);
        self.request_sender.send(client_req).await.unwrap();

        response_recv.recv().await.unwrap()
    }
}
