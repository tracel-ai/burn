use std::{collections::HashMap, marker::PhantomData, net::SocketAddr, sync::Arc};

use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{self, WebSocket},
    },
    response::IntoResponse,
    routing::any,
};
use burn_tensor::{TensorData, backend::Backend};
use futures_util::{SinkExt, StreamExt};
use tokio::{net::TcpStream, sync::Mutex};
use tokio::{runtime::Runtime, sync::Notify};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{self, protocol::WebSocketConfig},
};
use tokio_util::sync::CancellationToken;
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::{
    Layer, filter::filter_fn, layer::SubscriberExt, registry, util::SubscriberInitExt,
};

use crate::global::shared::NodeAddress;

#[derive(Clone)]
pub struct TensorDataClient<B: Backend> {
    state: Arc<TensorDataService<B>>,
}

type WebSocketStreamType = WebSocketStream<MaybeTlsStream<TcpStream>>;

struct TensorDataService<B: Backend> {
    /// Maps tensor IDs to their exposed state.
    pub exposed_tensors: Mutex<HashMap<u32, TensorExposeState>>,
    /// Maps node addresses to their WebSocket streams.
    pub ws_streams: Mutex<HashMap<NodeAddress, Arc<Mutex<WebSocketStreamType>>>>,
    /// Notify when a new tensor is exposed.
    pub new_tensor_notify: Arc<Notify>,

    cancel_token: CancellationToken,

    _phantom_data: PhantomData<B>,
}

pub struct TensorExposeState {
    /// The bytes of the tensor data
    pub bytes: bytes::Bytes,
    /// How many times the tensor will be downloaded
    pub max_downloads: u32,
    /// How man times the tensor has been downloaded
    pub cur_download_count: u32,
    /// Unique identifier between two nodes for the transfer of a tensor.
    pub transfer_id: u32,
}

impl<B: Backend> TensorDataClient<B> {
    pub fn new(runtime: &Runtime, cancel_token: CancellationToken, data_server_port: u16) -> Self {
        let state = Arc::new(TensorDataService::new(cancel_token.clone()));
        runtime.spawn(Self::start(state.clone(), cancel_token, data_server_port));

        Self { state }
    }

    /// Start the server on the given address.
    /// This will block until the server is stopped with the `cancel_token`.
    async fn start(state: Arc<TensorDataService<B>>, cancel_token: CancellationToken, port: u16) {
        // TODO abstract logging code
        let layer = tracing_subscriber::fmt::layer()
            .with_filter(LevelFilter::INFO)
            .with_filter(filter_fn(|m| {
                if let Some(path) = m.module_path() {
                    // The wgpu crate is logging too much, so we skip `info` level.
                    if path.starts_with("wgpu") && *m.level() >= Level::INFO {
                        return false;
                    }
                }
                true
            }));

        // If we start multiple servers in the same process, this will fail, it's ok
        let _ = registry().with(layer).try_init();

        let address = format!("0.0.0.0:{port}");
        log::info!("Start data server {address}");

        // build our application with some routes
        let app: Router = Router::new()
            .route("/data", any(Self::handler_data))
            .with_state(state.clone());

        let cancel_token = cancel_token.clone();
        let shutdown = async move {
            cancel_token.cancelled().await;
        };

        // run it with hyper
        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown)
        .await
        .unwrap();
    }

    async fn handler_data(
        ws: WebSocketUpgrade,
        State(state): State<Arc<TensorDataService<B>>>,
    ) -> impl IntoResponse {
        ws.on_upgrade(async move |socket| {
            state.handle_socket(socket).await;
        })
    }

    /// Exposes a tensor to the data server, allowing it to be downloaded by other nodes.
    pub(crate) async fn expose(
        &self,
        tensor: <B as Backend>::FloatTensorPrimitive,
        max_downloads: u32,
        transfer_id: u32,
    ) {
        self.state.expose(tensor, max_downloads, transfer_id).await
    }

    /// Downloads a tensor that is exposed on another server. Requires a Tokio 1.x runtime
    pub(crate) async fn download_next_tensor(
        &self,
        remote: &NodeAddress,
        tensor_id: u32,
    ) -> Option<TensorData> {
        self.state.download_next_tensor(remote, tensor_id).await
    }

    pub(crate) async fn close(&mut self) {
        self.state.close().await;
    }
}

impl<B: Backend> TensorDataService<B> {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self {
            exposed_tensors: Mutex::new(HashMap::new()),
            ws_streams: Mutex::new(HashMap::new()),
            new_tensor_notify: Arc::new(Notify::new()),
            cancel_token,
            _phantom_data: PhantomData::<B>,
        }
    }

    /// Exposes a tensor to the data server, allowing it to be downloaded by other nodes.
    pub(crate) async fn expose(
        &self,
        tensor: <B as Backend>::FloatTensorPrimitive,
        max_downloads: u32,
        transfer_id: u32,
    ) {
        let data = B::float_into_data(tensor).await;
        let bytes: bytes::Bytes = rmp_serde::to_vec(&data).unwrap().into();
        let mut exposed_tensors = self.exposed_tensors.lock().await;
        exposed_tensors.insert(
            transfer_id,
            TensorExposeState {
                bytes,
                max_downloads,
                cur_download_count: 0,
                transfer_id,
            },
        );
        core::mem::drop(exposed_tensors);
        self.new_tensor_notify.notify_waiters();
    }

    pub(crate) async fn close(&self) {
        // Send a closing message to every open WebSocket stream
        let reason = "Peer is closing".to_string();

        let mut ws_streams = self.ws_streams.lock().await;
        for (_, stream) in ws_streams.drain() {
            let mut stream = stream.lock().await;

            stream
                .send(tungstenite::Message::Close(Some(
                    tungstenite::protocol::CloseFrame {
                        code: tungstenite::protocol::frame::coding::CloseCode::Normal,
                        reason: reason.clone().into(),
                    },
                )))
                .await
                .expect("Failed to send Close message");
        }
    }

    /// Downloads a tensor that is exposed on another server. Requires a Tokio 1.x runtime
    /// TODO rename
    pub(crate) async fn download_next_tensor(
        &self,
        remote: &NodeAddress,
        tensor_id: u32,
    ) -> Option<TensorData> {
        log::info!("Downloading next tensor from {:?}", remote.0.as_str());

        let stream = self.get_data_stream(remote).await;
        let mut stream = stream.lock().await;

        // Send the download request with the download id
        let bytes: bytes::Bytes = rmp_serde::to_vec(&tensor_id).unwrap().into();
        stream
            .send(tungstenite::Message::Binary(bytes))
            .await
            .expect("Failed to send download id");

        if let Some(msg) = stream.next().await {
            let msg = match msg {
                Ok(msg) => msg,
                Err(err) => {
                    panic!(
                        "An error happened while receiving messages from the websocket: {err:?}"
                    );
                }
            };

            match msg {
                tungstenite::Message::Binary(bytes) => {
                    let data: TensorData = rmp_serde::from_slice(&bytes)
                        .expect("Can deserialize messages from the websocket.");
                    return Some(data);
                }
                tungstenite::Message::Close(_) => {
                    log::warn!("Closed connection");
                    return None;
                }
                _ => panic!("Unsupported websocket message: {msg:?}"),
            };
        }

        None
    }

    /// Get the WebSocket stream for the given address, or create a new one if it doesn't exist.
    async fn get_data_stream(
        &self,
        address: &NodeAddress,
    ) -> Arc<Mutex<WebSocketStream<MaybeTlsStream<TcpStream>>>> {
        let mut ws_streams = self.ws_streams.lock().await;
        match ws_streams.get(address) {
            Some(stream) => stream.clone(),
            None => {
                // Open a new WebSocket connection to the address
                let address_request = format!("{}/{}", address.0.as_str(), "data");
                const MB: usize = 1024 * 1024;
                let (stream, _) = connect_async_with_config(
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

                let stream = Arc::new(Mutex::new(stream));
                ws_streams.insert(address.clone(), stream.clone());

                stream
            }
        }
    }

    /// Handle incoming WebSocket connections for downloading tensors.
    pub async fn handle_socket(&self, mut socket: WebSocket) {
        log::info!("[Data Handler] New connection for download.");

        while !self.cancel_token.is_cancelled() {
            if let Some(msg) = socket.next().await {
                let msg = msg.unwrap_or_else(|err| {
                    panic!("Failed to receive message from websocket: {err:?}");
                });

                match msg {
                    ws::Message::Binary(bytes) => {
                        let tensor_id: u32 = rmp_serde::from_slice(&bytes)
                            .expect("Can deserialize messages from the websocket.");

                        let bytes = self.get_exposed_tensor_bytes(tensor_id).await.unwrap();

                        socket.send(ws::Message::Binary(bytes)).await.unwrap();
                    }
                    ws::Message::Close(_) => {
                        eprintln!("Closed connection");
                        return;
                    }
                    _ => panic!("Unsupported websocket message: {msg:?}"),
                };
            }
        }
        eprintln!("[Data Service] Closing connection for download.");
    }

    /// Get the requested exposed tensor data, and update download counter
    async fn get_exposed_tensor_bytes(&self, tensor_id: u32) -> Option<bytes::Bytes> {
        loop {
            {
                let mut exposed_tensors = self.exposed_tensors.lock().await;
                // take the tensor out of the hashmap while we download
                if let Some(mut exposed_state) = exposed_tensors.remove(&tensor_id) {
                    exposed_state.cur_download_count += 1;
                    let bytes = if exposed_state.cur_download_count == exposed_state.max_downloads {
                        exposed_state.bytes
                    } else {
                        let bytes = exposed_state.bytes.clone();
                        exposed_tensors.insert(tensor_id, exposed_state);
                        bytes
                    };
                    return Some(bytes);
                }
            }
            // No matching tensor, wait for a new one to come in.
            self.new_tensor_notify.notified().await;
        }
    }
}
