use std::{collections::HashMap, marker::PhantomData, sync::Arc, time::Duration};

use burn_communication::{Address, CommunicationChannel, Message, ProtocolClient};
use tokio::{
    runtime::Runtime,
    sync::{
        Mutex,
        mpsc::{Receiver, Sender},
    },
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use crate::global::shared::{
    CollectiveMessage, CollectiveMessageResponse, GlobalCollectiveError, RemoteRequest,
    RemoteResponse, RequestId, SessionId,
};

/// Worker that handles communication with the orchestrator for global collective operations.
pub(crate) struct GlobalClientWorker<P: ProtocolClient> {
    handle: Option<JoinHandle<Result<(), GlobalCollectiveError>>>,
    cancel_token: CancellationToken,
    request_sender: Sender<ClientRequest>,
    _phantom_data: PhantomData<P>,
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
    pub(crate) fn new(request: RemoteRequest, callback: Sender<RemoteResponse>) -> Self {
        Self { request, callback }
    }
}

impl<C: ProtocolClient> GlobalClientWorker<C> {
    /// Create a new global client worker and start the tasks.
    pub(crate) fn new(
        runtime: &Runtime,
        cancel_token: CancellationToken,
        global_address: &Address,
    ) -> Self {
        let (request_sender, request_recv) = tokio::sync::mpsc::channel::<ClientRequest>(10);

        let state = Arc::new(Mutex::new(GlobalClientWorkerState::new()));

        let handle = runtime.spawn(Self::start(
            state,
            cancel_token.clone(),
            global_address.clone(),
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
        global_address: Address,
        request_recv: Receiver<ClientRequest>,
    ) -> Result<(), GlobalCollectiveError> {
        // Init the connection.
        let (request, response) = Self::init_connection(&global_address).await?;

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

        if let Err(e) = response_handle.await {
            log::error!("Response handler failed: {e:?}");
        }
        if let Err(e) = request_handle.await {
            log::error!("Request handler failed: {e:?}");
        }

        Ok(())
    }

    async fn init_connection(
        address: &Address,
    ) -> Result<(C::Channel, C::Channel), GlobalCollectiveError> {
        let session_id = SessionId::new();

        let stream_request = tokio::spawn(Self::connect_with_retry(
            address.clone(),
            "request",
            std::time::Duration::from_secs(1),
            None,
            session_id,
        ));
        let stream_response = tokio::spawn(Self::connect_with_retry(
            address.clone(),
            "response",
            std::time::Duration::from_secs(1),
            None,
            session_id,
        ));

        let Ok(Some(request)) = stream_request.await else {
            return Err(GlobalCollectiveError::OrchestratorUnreachable);
        };
        let Ok(Some(response)) = stream_response.await else {
            return Err(GlobalCollectiveError::OrchestratorUnreachable);
        };

        Ok((request, response))
    }

    /// Connect with websocket with retries.
    async fn connect_with_retry(
        address: Address,
        route: &str,
        retry_pause: Duration,
        retry_max: Option<u32>,
        session_id: SessionId,
    ) -> Option<C::Channel> {
        let mut retries = 0;
        loop {
            if let Some(max) = retry_max
                && retries >= max
            {
                log::warn!("Failed to connect to {address} after {max} retries.");
                return None;
            }

            // Try to connect to the request address.
            println!("Connecting to {address} ...");
            let result = C::connect(address.clone(), route).await;

            if let Some(mut stream) = result {
                let init_msg = CollectiveMessage::Init(session_id);
                let bytes: bytes::Bytes = rmp_serde::to_vec(&init_msg).unwrap().into();
                stream
                    .send(Message::new(bytes))
                    .await
                    .expect("Can send the init message on the websocket.");
                return Some(stream);
            }

            println!("Failed to connect to {address}, retrying... Attempt #{retries}");
            tokio::time::sleep(retry_pause).await;
            retries += 1;
        }
    }

    /// Unregister the worker and close the connection.
    pub(crate) async fn close_connection(&mut self) -> Result<(), GlobalCollectiveError> {
        if let Some(handle) = self.handle.take() {
            // Un-register from server
            let req = RemoteRequest::Finish;
            let resp = self.request(req).await;
            if resp != RemoteResponse::FinishAck {
                log::error!("Requested to finish, did not get FinishAck; got {resp:?}");
                return Err(GlobalCollectiveError::WrongOrchestratorResponse);
            }

            self.cancel_token.cancel();

            if let Err(e) = handle.await.unwrap() {
                log::error!("Connection error {e:?}");
            }
        }

        Ok(())
    }

    async fn response_loader(
        state: Arc<Mutex<GlobalClientWorkerState>>,
        mut stream_response: C::Channel,
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
                            log::error!("Error receiving message from websocket: {err:?}");
                            break;
                        }
                        Ok(response) => {
                            let Some(response) = response else {
                                log::warn!("Closed connection");
                                break;
                            };

                            let response: CollectiveMessageResponse = rmp_serde::from_slice(&response.data)
                                .expect("Can deserialize messages from the websocket.");
                            let state_resp = state.lock().await;
                            let response_callback = state_resp
                                .requests
                                .get(&response.request_id)
                                .expect("Got a response to an unknown request");
                            response_callback.send(response.content).await.unwrap();
                        }
                    }
                }
            }
        }

        log::info!("Worker closing connection");
        stream_response
            .close()
            .await
            .expect("Can close the websocket stream.");
    }

    async fn request_sender(
        mut request_recv: Receiver<ClientRequest>,
        worker: Arc<Mutex<GlobalClientWorkerState>>,
        mut stream_request: C::Channel,
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

                    let request = CollectiveMessage::Request(id, request.request);

                    let bytes = rmp_serde::to_vec::<CollectiveMessage>(&request)
                        .expect("Can serialize tasks to bytes.")
                        .into();
                    stream_request
                        .send(Message::new(bytes))
                        .await
                        .expect("Can send the message on the websocket.");
                }
            }
        }

        log::info!("Worker closing connection");
        stream_request
            .close()
            .await
            .expect("Can send the close message on the websocket.");
    }

    pub(crate) async fn request(&self, req: RemoteRequest) -> RemoteResponse {
        let (callback, mut response_recv) = tokio::sync::mpsc::channel::<RemoteResponse>(10);
        let client_req = ClientRequest::new(req, callback);
        self.request_sender.send(client_req).await.unwrap();

        response_recv.recv().await.unwrap()
    }
}
