use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use burn_tensor::backend::Backend;
use tokio::sync::{
    Mutex,
    mpsc::{Sender},
};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{self, Message, protocol::WebSocketConfig},
};

use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;

use crate::{
    global::shared::{MessageResponse, NodeId, RemoteRequest, RemoteResponse, RequestId, SessionId}, GlobalRegisterParams
};

struct ClientRequest {
    req: RemoteRequest,
    callback: Sender<RemoteResponse>,
}

pub(crate) struct GlobalCollectiveClient<B: Backend> {
    request_sender: Sender<ClientRequest>,
    _phantom_data: PhantomData<B>,
}

struct ClientWorker {
    requests: HashMap<RequestId, Sender<RemoteResponse>>
}

impl ClientWorker {
    fn new() -> Self {
        Self {
            requests: HashMap::new(),
        }
    }
}

impl<B: Backend> GlobalCollectiveClient<B> {
    pub fn start(
        server_address: String,
    ) -> Self {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap(),
        );

        let (request_sender, mut request_recv) = tokio::sync::mpsc::channel::<ClientRequest>(10);
        let worker = ClientWorker::new();
        let worker: Arc<Mutex<ClientWorker>> = Arc::new(tokio::sync::Mutex::new(worker));

        let address_request = format!("{}/{}", server_address.as_str(), "request");
        let address_response = format!("{}/{}", server_address.as_str(), "response");

        const MB: usize = 1024 * 1024;

        let state = worker.clone();
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

            // Init the connection.
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

            // Websocket async worker loading responses from the server.
            let state_resp = state.clone();
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
                            let response: MessageResponse = rmp_serde::from_slice(&bytes)
                                .expect("Can deserialize messages from the websocket.");
                            let state_resp = state_resp.lock().await;
                            let response_callback =
                                state_resp.requests.get(&response.id).expect(&format!(
                                    "Got a response to an unknown request (with ID: {:?})",
                                    response.id
                                ));
                            response_callback.send(response.content).await.unwrap();
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
                while let Some(request) = request_recv.recv().await {
                    let id = RequestId::new();

                    let mut state = state.lock().await;
                    state.requests.insert(id, request.callback);

                    let request = crate::global::shared::Message::Request(id, request.req);

                    let bytes = rmp_serde::to_vec::<crate::global::shared::Message>(&request)
                        .expect("Can serialize tasks to bytes.")
                        .into();
                    stream_request
                        .send(Message::Binary(bytes))
                        .await
                        .expect("Can send the message on the websocket.");
                }
            });
        });

        Self {
            request_sender,
            _phantom_data: PhantomData,
        }
    }

    pub fn new(server_address: String) -> Self {
        Self::start(server_address)
    }

    pub async fn request(&mut self, req: RemoteRequest) -> RemoteResponse {
        let (callback, mut response_recv) = tokio::sync::mpsc::channel::<RemoteResponse>(10);
        let client_req = ClientRequest { req, callback };
        self.request_sender.send(client_req).await.unwrap();

        response_recv.recv().await.unwrap()
    }

    pub async fn register(&mut self, node_id: NodeId, params: GlobalRegisterParams) {
        let req = RemoteRequest::Register { node_id, num_nodes: params.num_nodes };
        let resp = self.request(req).await;
        if resp != RemoteResponse::RegisterAck {
            panic!("The response to a register request should be a RegisterAck, not {:?}", resp);
        }
    }

    pub async fn aggregate(&self, _tensor: B::FloatTensorPrimitive) {
        todo!();
        // let req = RemoteRequest::Aggregate { tensor: todo!(), params: () };
        // let resp = self.request(req).await;
        // if resp != RemoteResponse::RegisterAck {
        //     panic!("The response to a register request should be a RegisterAck, not {:?}", resp);
        // }
    }
}
