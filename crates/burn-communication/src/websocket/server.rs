use std::net::SocketAddr;

use crate::{
    base::{CommunicationChannel, CommunicationError, Message, ProtocolServer},
    util::init_logging,
};
use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{self, WebSocket},
    },
    routing::get,
};
use futures::StreamExt;

#[derive(Clone, Debug)]
pub struct WsServer {
    port: u16,
    router: Router<()>,
}

pub struct WsServerChannel {
    inner: WebSocket,
}

impl WsServer {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            router: Router::new(),
        }
    }
}

impl ProtocolServer for WsServer {
    type Channel = WsServerChannel;
    type Error = WsServerError;

    async fn serve<F>(self, shutdown: F) -> Result<(), Self::Error>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        init_logging();

        let address = format!("0.0.0.0:{}", self.port);
        log::info!("Starting server {address}");

        let listener = tokio::net::TcpListener::bind(address).await?;

        axum::serve(
            listener,
            self.router
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown)
        .await?;

        Ok(())
    }

    fn route<C, Fut>(mut self, path: &str, callback: C) -> Self
    where
        C: FnOnce(WsServerChannel) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        // Format path: should start with a /
        let path = if path.starts_with("/") {
            path.to_owned()
        } else {
            format!("/{path}")
        };

        let method = get(|ws: WebSocketUpgrade, _: State<()>| async {
            ws.on_upgrade(async move |socket| {
                callback(WsServerChannel { inner: socket }).await;
            })
        });

        self.router = self.router.route(&path, method);

        self
    }
}

impl CommunicationChannel for WsServerChannel {
    type Error = WsServerError;

    async fn send(&mut self, message: Message) -> Result<(), WsServerError> {
        self.inner.send(ws::Message::Binary(message.data)).await?;

        Ok(())
    }

    async fn recv(&mut self) -> Result<Option<Message>, WsServerError> {
        match self.inner.next().await {
            Some(next) => match next {
                Ok(ws::Message::Binary(data)) => Ok(Some(Message { data })),
                Ok(ws::Message::Close(_close_frame)) => Ok(None),
                Err(err) => Err(WsServerError::Axum(err)),
                msg => Err(WsServerError::UnknownMessage(format!("{msg:?}"))),
            },
            None => todo!(),
        }
    }

    async fn close(&mut self) -> Result<(), WsServerError> {
        let reason = "Peer is closing".to_string();

        self.inner
            .send(ws::Message::Close(Some(ws::CloseFrame {
                code: 1000, // code: Normal
                reason: reason.clone().into(),
            })))
            .await?;

        Ok(())
    }
}

#[derive(Debug)]
pub enum WsServerError {
    Io(std::io::Error),
    Axum(axum::Error),
    UnknownMessage(String),
    Other(String),
}

impl CommunicationError for WsServerError {}

impl From<std::io::Error> for WsServerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<axum::Error> for WsServerError {
    fn from(err: axum::Error) -> Self {
        Self::Axum(err)
    }
}
