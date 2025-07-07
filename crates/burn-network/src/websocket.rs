use std::net::SocketAddr;

use crate::network::{NetworkClient, NetworkMessage, NetworkServer, NetworkStream};
use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{self, WebSocket},
    },
    routing::get,
};
use burn_common::future::DynFut;
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

pub struct WsClient;
impl NetworkClient for WsClient {
    type ClientStream = WsClientStream;
    fn connect(address: String) -> DynFut<Option<WsClientStream>> {
        Box::pin(connect_ws(address))
    }
}

async fn connect_ws(address: String) -> Option<WsClientStream> {
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

    Some(WsClientStream { inner: stream })
}

pub struct WsServer<S: Clone + Send + Sync + 'static> {
    port: u16,
    router: Router<S>,
}

#[derive(Debug)]
pub enum WsNetworkError {
    Axum(axum::Error),
    Tungstenite(tungstenite::Error),
    UnknownMessage(String),
}

pub struct WsServerStream {
    inner: WebSocket,
}

pub struct WsClientStream {
    inner: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

impl<S: Clone + Send + Sync + 'static> NetworkServer for WsServer<S> {
    type State = S;
    type ServerStream = WsServerStream;

    fn new(port: u16) -> Self {
        Self {
            port,
            router: Router::new(),
        }
    }

    async fn serve<F>(self, state: S, shutdown: F)
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

    fn route<C, Fut>(mut self, path: &str, callback: C) -> Self
    where
        C: FnOnce(S, WsServerStream) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let method = get(|ws: WebSocketUpgrade, State(state): State<S>| async {
            ws.on_upgrade(async move |socket| {
                callback(state, WsServerStream { inner: socket }).await;
            })
        });

        self.router = self.router.route(path, method);

        self
    }
}

impl NetworkStream for WsServerStream {
    type Error = WsNetworkError;

    async fn send(&mut self, bytes: bytes::Bytes) -> Result<(), WsNetworkError> {
        self.inner
            .send(ws::Message::Binary(bytes))
            .await
            .map_err(WsNetworkError::Axum)
    }

    async fn recv(&mut self) -> Result<Option<NetworkMessage>, WsNetworkError> {
        match self.inner.next().await {
            Some(next) => match next {
                Ok(ws::Message::Binary(data)) => Ok(Some(NetworkMessage { data })),
                Ok(ws::Message::Close(_close_frame)) => Ok(None),
                Err(err) => Err(WsNetworkError::Axum(err)),
                msg => Err(WsNetworkError::UnknownMessage(format!("{:?}", msg))),
            },
            None => todo!(),
        }
    }

    async fn close(&mut self) -> Result<(), WsNetworkError> {
        let reason = "Peer is closing".to_string();

        self.inner
            .send(ws::Message::Close(Some(ws::CloseFrame {
                code: 1000, // code: Normal
                reason: reason.clone().into(),
            })))
            .await
            .map_err(WsNetworkError::Axum)
    }
}

impl NetworkStream for WsClientStream {
    type Error = WsNetworkError;

    async fn send(&mut self, bytes: bytes::Bytes) -> Result<(), WsNetworkError> {
        self.inner
            .send(tungstenite::Message::Binary(bytes))
            .await
            .map_err(WsNetworkError::Tungstenite)
    }

    async fn recv(&mut self) -> Result<Option<NetworkMessage>, WsNetworkError> {
        match self.inner.next().await {
            Some(next) => match next {
                Ok(tungstenite::Message::Binary(data)) => Ok(Some(NetworkMessage { data })),
                Ok(tungstenite::Message::Close(_close_frame)) => Ok(None),
                Err(err) => Err(WsNetworkError::Tungstenite(err)),
                msg => Err(WsNetworkError::UnknownMessage(format!("{:?}", msg))),
            },
            None => todo!(),
        }
    }

    async fn close(&mut self) -> Result<(), WsNetworkError> {
        let reason = "Peer is closing".to_string();

        self.inner
            .send(tungstenite::Message::Close(Some(
                tungstenite::protocol::CloseFrame {
                    code: tungstenite::protocol::frame::coding::CloseCode::Normal,
                    reason: reason.clone().into(),
                },
            )))
            .await
            .map_err(WsNetworkError::Tungstenite)
    }
}
