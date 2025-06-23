use burn_tensor::backend::Backend;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
use tokio::sync::{Mutex, mpsc::Sender};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{self, Message, protocol::WebSocketConfig},
};

use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;

use crate::{
    GlobalAggregateParams, GlobalRegisterParams,
    global::{
        client::data_server::{TensorDataClient, download_next_tensor},
        shared::{
            CentralizedAggregateStrategy::{Central, Peripheral},
            MessageResponse, NodeAddress, NodeId, RemoteRequest, RemoteResponse, RequestId,
            SessionId,
        },
    },
};

struct ClientRequest {
    request: RemoteRequest,
    callback: Sender<RemoteResponse>,
}

impl ClientRequest {
    fn new(request: RemoteRequest, callback: Sender<RemoteResponse>) -> Self {
        Self { request, callback }
    }
}

pub(crate) struct GlobalCollectiveClient<B: Backend> {
    request_sender: Sender<ClientRequest>,
    data_client: TensorDataClient<B>,
    data_client_address: Arc<NodeAddress>,
    _phantom_data: PhantomData<B>,
}

struct ClientWorker {
    requests: HashMap<RequestId, Sender<RemoteResponse>>,
}

impl ClientWorker {
    fn new() -> Self {
        Self {
            requests: HashMap::new(),
        }
    }
}

impl<B: Backend> GlobalCollectiveClient<B> {
    pub fn start(server_address: &str, client_address: &str, data_server_port: u16) -> Self {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap(),
        );

        let (request_sender, mut request_recv) = tokio::sync::mpsc::channel::<ClientRequest>(10);
        let worker = ClientWorker::new();
        let worker: Arc<Mutex<ClientWorker>> = Arc::new(tokio::sync::Mutex::new(worker));

        let address_request = format!("{}/{}", server_address, "request");
        let address_response = format!("{}/{}", server_address, "response");

        let data_client = TensorDataClient::<B>::new();

        const MB: usize = 1024 * 1024;

        let worker_clone = worker.clone();
        let data_client_clone = data_client.clone();
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
            let worker = worker_clone;
            let worker_clone = worker.clone();
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
                            let state_resp = worker_clone.lock().await;
                            let response_callback = state_resp
                                .requests
                                .get(&response.id)
                                .expect("Got a response to an unknown request");
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
            });

            // Server for transfering tensors to and from other nodes
            tokio::spawn(data_client_clone.start(data_server_port));
        });

        Self {
            request_sender,
            data_client,
            data_client_address: Arc::new(NodeAddress(client_address.to_owned())),
            _phantom_data: PhantomData,
        }
    }

    pub fn new(server_address: &str, client_address: &str, data_server_port: u16) -> Self {
        Self::start(server_address, client_address, data_server_port)
    }

    pub async fn request_with_callback(&mut self, req: RemoteRequest) -> RemoteResponse {
        let (callback, mut response_recv) = tokio::sync::mpsc::channel::<RemoteResponse>(10);
        let client_req = ClientRequest::new(req, callback);
        self.request_sender.send(client_req).await.unwrap();

        response_recv.recv().await.unwrap()
    }

    pub async fn register(&mut self, node_id: NodeId, params: GlobalRegisterParams) {
        let node_addr = self.data_client_address.as_ref().clone();
        let req = RemoteRequest::Register {
            node_id,
            node_addr,
            num_nodes: params.num_nodes,
        };
        let resp = self.request_with_callback(req).await;
        if resp != RemoteResponse::RegisterAck {
            panic!(
                "The response to a register request should be a RegisterAck, not {:?}",
                resp
            );
        }
    }

    pub async fn aggregate(
        &mut self,
        tensor: &B::FloatTensorPrimitive,
        params: GlobalAggregateParams,
        device: &B::Device,
    ) -> B::FloatTensorPrimitive {
        let req = RemoteRequest::Aggregate { params };
        let resp = self.request_with_callback(req).await;
        let strategy = match resp {
            RemoteResponse::AggregateStrategy(strategy) => strategy,
            RemoteResponse::Error(err) => panic!("Global collective server error: {err}"),
            resp => panic!(
                "The response to a register request should be a RegisterAck, not {:?}",
                resp
            ),
        };

        match strategy {
            Central { other_nodes } => {
                // download tensors from other nodes
                let downloads = other_nodes.iter().map(download_next_tensor);

                let results = futures::future::join_all(downloads).await;
                let mut results = results.into_iter().map(|res| {
                    let data = res.unwrap();
                    B::float_from_data(data, device)
                });

                // sum
                let mut first_tensor = results.next().unwrap();
                for res in results {
                    first_tensor = B::float_add(first_tensor, res);
                }

                // Expose result
                self.data_client.expose(&first_tensor, 1).await;

                first_tensor
            }
            Peripheral { central_node } => {
                // Expose input
                self.data_client.expose(tensor, 1).await;

                // Download result
                let data = download_next_tensor(&central_node).await.unwrap();

                B::float_from_data(data, device)
            }
        }
    }
}
