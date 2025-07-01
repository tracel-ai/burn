use burn_tensor::backend::Backend;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
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

use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;

use crate::{
    GlobalAllReduceParams, GlobalRegisterParams,
    global::{
        client::data_server::{TensorDataClient, download_next_tensor},
        shared::{
            CentralizedAllReduceStrategy::{Central, Peripheral},
            MessageResponse, NodeAddress, RemoteRequest, RemoteResponse, RequestId, SessionId,
        },
    },
};

#[derive(Debug)]
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
    cancel_token: CancellationToken,
    worker_runtime: Runtime,
    worker_handle: Option<JoinHandle<()>>,
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
    pub fn new(server_address: &str, client_address: &str, data_server_port: u16) -> Self {
        let (request_sender, request_recv) = tokio::sync::mpsc::channel::<ClientRequest>(10);
        let worker = ClientWorker::new();
        let worker: Arc<Mutex<ClientWorker>> = Arc::new(tokio::sync::Mutex::new(worker));

        let data_client = TensorDataClient::<B>::new();

        let address_request = format!("{}/{}", server_address, "request");
        let address_response = format!("{}/{}", server_address, "response");

        let cancel_token = CancellationToken::new();

        let data_client_clone = data_client.clone();

        let _runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let worker_handle = _runtime.spawn({
            let cancel_token = cancel_token.clone();
            async move {
                // Init the connection.
                let (request, response) =
                    Self::init_connection(address_request, address_response).await;

                // Websocket async worker loading responses from the server.
                tokio::spawn(Self::response_loader(
                    worker.clone(),
                    response,
                    cancel_token.clone(),
                ));

                // Channel async worker sending operations to the server.
                tokio::spawn(Self::request_sender(
                    request_recv,
                    worker,
                    request,
                    cancel_token.clone(),
                ));

                // Server for transfering tensors to and from other nodes
                tokio::spawn(data_client_clone.start(data_server_port, cancel_token));
            }
        });

        Self {
            request_sender,
            data_client,
            data_client_address: Arc::new(NodeAddress(client_address.to_owned())),
            cancel_token,
            worker_runtime: _runtime,
            worker_handle: Some(worker_handle),
            _phantom_data: PhantomData,
        }
    }

    async fn init_connection(
        address_request: String,
        address_response: String,
    ) -> (
        WebSocketStream<MaybeTlsStream<TcpStream>>,
        WebSocketStream<MaybeTlsStream<TcpStream>>,
    ) {
        const MB: usize = 1024 * 1024;

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

    fn close_connection(&mut self) {
        self.cancel_token.cancel();
        let handle = self.worker_handle.take().unwrap();
        self.worker_runtime.block_on(handle).unwrap();
    }

    async fn response_loader(
        worker: Arc<Mutex<ClientWorker>>,
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
                        let state_resp = worker.lock().await;
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
        }
    }

    async fn request_sender(
        mut request_recv: Receiver<ClientRequest>,
        worker: Arc<Mutex<ClientWorker>>,
        mut stream_request: WebSocketStream<MaybeTlsStream<TcpStream>>,
        cancel_token: CancellationToken,
    ) {
        while !cancel_token.is_cancelled() {
            if let Some(request) = request_recv.recv().await {
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
        }
    }

    pub async fn request_with_callback(&mut self, req: RemoteRequest) -> RemoteResponse {
        let (callback, mut response_recv) = tokio::sync::mpsc::channel::<RemoteResponse>(10);
        let client_req = ClientRequest::new(req, callback);
        self.request_sender.send(client_req).await.unwrap();

        response_recv.recv().await.unwrap()
    }

    pub async fn register(&mut self, node_id: u32, params: GlobalRegisterParams) {
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

    pub async fn all_reduce(
        &mut self,
        tensor: B::FloatTensorPrimitive,
        params: GlobalAllReduceParams,
        device: &B::Device,
    ) -> B::FloatTensorPrimitive {
        let req = RemoteRequest::AllReduce { params };
        let resp = self.request_with_callback(req).await;
        let strategy = match resp {
            RemoteResponse::AllReduceStrategy(strategy) => strategy,
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
                let results = results.into_iter().map(|res| {
                    let data = res.unwrap();
                    B::float_from_data(data, device)
                });

                // sum
                let mut sum = tensor;
                for res in results {
                    sum = B::float_add(sum, res);
                }

                // Expose result
                self.data_client
                    .expose(sum.clone(), other_nodes.len())
                    .await;

                sum
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

    pub async fn finish(&mut self) {
        // Un-register from server
        let req = RemoteRequest::Finish;
        let resp = self.request_with_callback(req).await;
        if resp != RemoteResponse::FinishAck {
            panic!("Requested to finish, did not get FinishAck; got {:?}", resp);
        }

        // Close worker thread
        self.close_connection();
    }
}

impl<B: Backend> Drop for GlobalCollectiveClient<B> {
    fn drop(&mut self) {
        eprintln!("Dropping Global Collective Client");
    }
}
