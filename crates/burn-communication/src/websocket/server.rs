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

    /// Serve on an already-bound listener instead of binding `0.0.0.0:port` ourselves.
    ///
    /// This is what [`serve`](ProtocolServer::serve) delegates to; it is exposed so a caller
    /// that needs the actual bound address — e.g. a test binding an ephemeral `:0` port and
    /// reading the port back from `listener.local_addr()` — can do so without a fixed port.
    pub async fn serve_on<F>(
        self,
        listener: tokio::net::TcpListener,
        shutdown: F,
    ) -> Result<(), WsServerError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        init_logging();

        // Report the address the listener actually bound to (resolves an ephemeral `:0` port to
        // the real one), so the log confirms the server is accepting connections and tells the
        // operator where.
        match listener.local_addr() {
            Ok(addr) => log::info!("Server started, listening on {addr}"),
            Err(err) => log::info!("Server started (could not resolve bound address: {err})"),
        }

        axum::serve(
            listener,
            self.router
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown)
        .await?;

        Ok(())
    }
}

impl ProtocolServer for WsServer {
    type Channel = WsServerChannel;
    type Error = WsServerError;

    async fn serve<F>(self, shutdown: F) -> Result<(), Self::Error>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let address = format!("0.0.0.0:{}", self.port);
        log::info!("Starting server {address}");

        let listener = tokio::net::TcpListener::bind(address).await?;

        self.serve_on(listener, shutdown).await
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
        // Keep reading until we get a data frame, a close, an error, or the stream ends.
        // Ping/Pong (keepalives) and Text frames must be skipped, not turned into errors: a
        // single keepalive ping would otherwise look like a fatal error to the consuming loop
        // and kill an otherwise-healthy connection. tungstenite already answers pings with
        // pongs at the protocol layer, so ignoring them here is safe.
        loop {
            match self.inner.next().await {
                Some(Ok(ws::Message::Binary(data))) => return Ok(Some(Message { data })),
                // A close frame is an orderly end-of-stream.
                Some(Ok(ws::Message::Close(_close_frame))) => return Ok(None),
                // Control/text frames carry no protocol payload: skip and keep reading.
                Some(Ok(ws::Message::Ping(_) | ws::Message::Pong(_) | ws::Message::Text(_))) => {
                    continue;
                }
                Some(Err(err)) => return Err(WsServerError::Axum(err)),
                // The stream is exhausted: the peer went away without a close frame (closed
                // tab, network drop, killed client). This is a common disconnect path, not an
                // error — treat it as a clean end-of-stream rather than panicking.
                None => return Ok(None),
            }
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
