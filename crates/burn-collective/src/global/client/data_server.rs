use std::{collections::VecDeque, marker::PhantomData, net::SocketAddr, sync::Arc};

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
use futures_util::StreamExt;
use std::sync::Mutex;
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{Message, protocol::WebSocketConfig},
};
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::{
    Layer, filter::filter_fn, layer::SubscriberExt, registry, util::SubscriberInitExt,
};

use crate::global::shared::NodeAddress;

/// Downloads a tensor that is exposed on another server. Requires a Tokio 1.x runtime
pub async fn download_next_tensor(remote: &NodeAddress) -> Option<TensorData> {
    log::info!("Downloading next tensor from {:?}", remote.0.as_str());
    let address_request = format!("{}/{}", remote.0.as_str(), "data");
    const MB: usize = 1024 * 1024;

    let (mut stream, _) = connect_async_with_config(
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

    if let Some(msg) = stream.next().await {
        let msg = match msg {
            Ok(msg) => msg,
            Err(err) => {
                panic!("An error happened while receiving messages from the websocket: {err:?}")
            }
        };

        match msg {
            Message::Binary(bytes) => {
                let data: TensorData = rmp_serde::from_slice(&bytes)
                    .expect("Can deserialize messages from the websocket.");
                return Some(data);
            }
            Message::Close(_) => {
                log::warn!("Closed connection");
                return None;
            }
            _ => panic!("Unsupported websocket message: {msg:?}"),
        };
    }

    None
}

#[derive(Clone)]
pub struct TensorDataClient<B: Backend> {
    state: Arc<TensorDataService>,
    _phantom_data: PhantomData<B>,
}

impl<B: Backend> TensorDataClient<B> {
    pub fn new() -> Self {
        Self {
            state: Arc::new(TensorDataService::new()),
            _phantom_data: PhantomData,
        }
    }

    /// Start the server on the given address.
    pub async fn start(self, port: u16) {
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
        let app = Router::new()
            .route("/data", any(Self::handler_data))
            .with_state(self.clone());

        // run it with hyper
        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    }

    async fn handler_data(ws: WebSocketUpgrade, State(state): State<Self>) -> impl IntoResponse {
        ws.on_upgrade(async move |socket| {
            state.state.handle_socket_data(socket).await;
        })
    }

    pub async fn expose(&mut self, tensor: &B::FloatTensorPrimitive, count: u32) {
        // TODO is cloning here ok?
        let data = B::float_into_data(tensor.clone()).await;
        let bytes: bytes::Bytes = rmp_serde::to_vec(&data).unwrap().into();

        let mut exposed_tensors = self.state.exposed_tensors.lock().unwrap();
        exposed_tensors.push_back(TensorExposeState {
            bytes,
            max_downloads: count,
            cur_download_count: 0,
        });
    }
}

impl<B: Backend> Default for TensorDataClient<B> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TensorExposeState {
    pub bytes: bytes::Bytes,
    pub max_downloads: u32,
    pub cur_download_count: u32,
}

pub struct TensorDataService {
    pub exposed_tensors: Mutex<VecDeque<TensorExposeState>>,
}

impl TensorDataService {
    pub fn new() -> Self {
        Self {
            exposed_tensors: Mutex::new(VecDeque::new()),
        }
    }

    pub async fn handle_socket_data(&self, mut socket: WebSocket) {
        log::info!("[Data Handler] New connection for download.");

        // TODO there should be counters for the different steps of collective operations
        // depending on the strategy

        // Get the requested exposed tensor data
        let bytes: bytes::Bytes = self.get_next_exposed_tensor_bytes().await;

        // Send tensor and increment its counter
        socket.send(ws::Message::Binary(bytes)).await.unwrap();
    }

    async fn get_next_exposed_tensor_bytes(&self) -> bytes::Bytes {
        loop {
            if let Ok(mut exposed_tensors) = self.exposed_tensors.try_lock() {
                // take the tensor out of the hashmap while we download
                if let Some(mut exposed_state) = exposed_tensors.pop_front() {
                    exposed_state.cur_download_count += 1;
                    if exposed_state.cur_download_count == exposed_state.max_downloads {
                        return exposed_state.bytes;
                    } else {
                        let bytes = exposed_state.bytes.clone();
                        exposed_tensors.push_front(exposed_state);
                        return bytes;
                    }
                } else {
                    //panic!("A tensor was requested (id: {id:?}) that isn't being served");
                }
            }
        }
    }
}

impl Default for TensorDataService {
    fn default() -> Self {
        Self::new()
    }
}
