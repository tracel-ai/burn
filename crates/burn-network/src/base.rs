use std::net::SocketAddr;

use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{self, WebSocket},
    },
    routing::get,
};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{self, protocol::WebSocketConfig},
};
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::{
    Layer, filter::filter_fn, layer::SubscriberExt, registry, util::SubscriberInitExt,
};

pub fn init_logging() {
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
}

pub struct Server<S: Clone + Send + Sync + 'static> {
    port: u16,
    router: Router<S>,
}

pub struct ServerNetworkStream {
    inner: WebSocket,
}

pub struct ClientNetworkStream {
    inner: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

pub struct NetworkMessage {
    pub data: bytes::Bytes,
}

#[derive(Debug)]
pub enum NetworkError {
    Axum(axum::Error),
    Tungstenite(tungstenite::Error),
    UnknownMessage(String),
}

impl<S: Clone + Send + Sync + 'static> Server<S> {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            router: Router::new(),
        }
    }

    pub async fn serve<F>(self, state: S, shutdown: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        init_logging();

        let address = format!("0.0.0.0:{}", self.port);
        log::info!("Start data server {address}");

        let app: Router = self.router.with_state(state.clone());

        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown)
        .await
        .unwrap();
    }

    pub fn route<C, Fut>(mut self, path: &str, callback: C) -> Self
    where
        C: FnOnce(S, ServerNetworkStream) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let method = get(|ws: WebSocketUpgrade, State(state): State<S>| async {
            ws.on_upgrade(async move |socket| {
                callback(state, ServerNetworkStream { inner: socket }).await;
            })
        });

        self.router = self.router.route(path, method);

        self
    }
}

impl ServerNetworkStream {
    pub async fn send(&mut self, bytes: bytes::Bytes) -> Result<(), NetworkError> {
        self.inner
            .send(ws::Message::Binary(bytes))
            .await
            .map_err(NetworkError::Axum)
    }

    pub async fn recv(&mut self) -> Result<Option<NetworkMessage>, NetworkError> {
        match self.inner.next().await {
            Some(next) => match next {
                Ok(ws::Message::Binary(data)) => Ok(Some(NetworkMessage { data })),
                Ok(ws::Message::Close(_close_frame)) => Ok(None),
                Err(err) => Err(NetworkError::Axum(err)),
                msg => Err(NetworkError::UnknownMessage(format!("{:?}", msg))),
            },
            None => todo!(),
        }
    }
}

impl ClientNetworkStream {
    pub async fn send(&mut self, bytes: bytes::Bytes) -> Result<(), NetworkError> {
        self.inner
            .send(tungstenite::Message::Binary(bytes))
            .await
            .map_err(NetworkError::Tungstenite)
    }

    pub async fn recv(&mut self) -> Result<Option<NetworkMessage>, NetworkError> {
        match self.inner.next().await {
            Some(next) => match next {
                Ok(tungstenite::Message::Binary(data)) => Ok(Some(NetworkMessage { data })),
                Ok(tungstenite::Message::Close(_close_frame)) => Ok(None),
                Err(err) => Err(NetworkError::Tungstenite(err)),
                msg => Err(NetworkError::UnknownMessage(format!("{:?}", msg))),
            },
            None => todo!(),
        }
    }

    pub async fn close(&mut self) -> Result<(), NetworkError> {
        let reason = "Peer is closing".to_string();

        self.inner
            .send(tungstenite::Message::Close(Some(
                tungstenite::protocol::CloseFrame {
                    code: tungstenite::protocol::frame::coding::CloseCode::Normal,
                    reason: reason.clone().into(),
                },
            )))
            .await
            .map_err(NetworkError::Tungstenite)
    }
}

pub async fn connect(address: String) -> Option<ClientNetworkStream> {
    // Open a new WebSocket connection to the address
    const MB: usize = 1024 * 1024;
    let (stream, _) = connect_async_with_config(
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
    .await
    .ok()?;

    Some(ClientNetworkStream { inner: stream })
}

pub async fn os_shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
