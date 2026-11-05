//! Serving over WebSocket.

use std::sync::Arc;

use burn_communication::{
    external_comm::{ExternalCommServer, ExternalCommService},
    websocket::{WebSocket, WsServer, WsServerChannel},
};
use burn_ir::BackendIr;
use tokio::net::TcpListener;

use super::transfer::WebSocketTransfer;
use crate::{
    Credential,
    server::{AuthorizationRequest, ClientId, ServeError, SessionSetup, pump::drive_session},
    transport::link::MAX_FRAME_SIZE,
};

/// How a server accepts clients over WebSocket: the port it listens on, on every interface.
///
/// WebSocket is unencrypted: a token stops stray clients on a trusted network, not someone reading
/// the traffic. Serve over Iroh to cross an untrusted one.
#[derive(Debug)]
pub struct WebSocketTransport {
    listen: Listen,
}

#[derive(Debug)]
enum Listen {
    Port(u16),
    Listener(std::net::TcpListener),
}

impl WebSocketTransport {
    /// Listen on TCP `port`, on every IPv4 interface.
    pub fn new(port: u16) -> Self {
        Self {
            listen: Listen::Port(port),
        }
    }

    /// Serve on a listener the caller bound, such as one on port 0 whose port it reads back
    /// before any client dials.
    pub fn from_listener(listener: std::net::TcpListener) -> Self {
        Self {
            listen: Listen::Listener(listener),
        }
    }

    /// Bind the listening socket, ready to serve on.
    pub(crate) async fn bind(self) -> Result<WebSocketListener, ServeError> {
        let listener = match self.listen {
            Listen::Port(port) => TcpListener::bind(("0.0.0.0", port)).await,
            Listen::Listener(listener) => listener
                .set_nonblocking(true)
                .and_then(|()| TcpListener::from_std(listener)),
        };
        listener.map(WebSocketListener).map_err(ServeError::bind)
    }
}

/// A bound WebSocket socket, which accepts clients once served.
pub(crate) struct WebSocketListener(TcpListener);

impl WebSocketListener {
    /// Serve until `setup`'s shutdown is cancelled, or the returned future is dropped.
    pub(crate) async fn serve<B: BackendIr>(
        self,
        setup: SessionSetup<B>,
    ) -> Result<(), ServeError> {
        let shutdown = setup.shutdown.clone();
        compute_server(setup)
            .serve_on(self.0, shutdown.cancelled_owned())
            .await
            .map_err(ServeError::transport)
    }
}

/// The compute node's routes: one full-duplex `/session` socket per session, driven by the shared
/// [`drive_session`] pump, and the tensor transfers between servers.
fn compute_server<B: BackendIr>(setup: SessionSetup<B>) -> WsServer {
    let external = Arc::new(ExternalCommService::<B, WebSocket>::new(
        setup.shutdown.clone(),
    ));
    let transfer = Arc::new(WebSocketTransfer {
        inner: external.clone(),
    });
    let sessions = Arc::new(setup.manager(transfer));
    let authorizer = setup.authorizer;
    let shutdown = setup.shutdown;

    // `serve_on` serves on the listener it is given; this port is never bound.
    WsServer::new(0)
        // Axum fixes a socket's limit for its life, so an `Init` meets its own limit once read.
        .route_with_max_message_size(
            "/session",
            MAX_FRAME_SIZE,
            move |channel: WsServerChannel| {
                let sessions = sessions.clone();
                let authorizer = authorizer.clone();
                let shutdown = shutdown.clone();
                async move {
                    let client = ClientId::WebSocket(channel.peer_addr());
                    let (sink, source) = channel.split();
                    let served = drive_session(source, sink, sessions, None, &shutdown, |init| {
                        authorizer.authorize(AuthorizationRequest {
                            client,
                            device_index: init.device_index,
                            credential: &Credential::from(init.authorization.as_slice()),
                        })
                    })
                    .await;
                    if let Err(err) = served {
                        log::warn!("Rejected or failed WebSocket remote session: {err}");
                    }
                }
            },
        )
        .route_external_comm(external)
}
