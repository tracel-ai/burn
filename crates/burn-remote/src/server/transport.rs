use burn_ir::BackendIr;

#[cfg(feature = "iroh")]
use crate::transport::iroh::{IrohListener, IrohTransport};
#[cfg(feature = "websocket")]
use crate::transport::websocket::{WebSocketListener, WebSocketTransport};

use super::{ServeError, SessionSetup};

/// How a server accepts clients, built with `into` from an [`IrohTransport`] or a
/// [`WebSocketTransport`].
///
/// Opaque, so a match on it cannot depend on which transport features the build enables.
#[derive(Debug)]
pub struct Transport(TransportKind);

#[derive(Debug)]
enum TransportKind {
    #[cfg(feature = "iroh")]
    Iroh(IrohTransport),
    #[cfg(feature = "websocket")]
    WebSocket(WebSocketTransport),
}

impl Transport {
    /// Bind the transport's socket or endpoint, the step a taken port or an invalid setting fails.
    pub(crate) async fn bind(self) -> Result<Listener, ServeError> {
        match self.0 {
            #[cfg(feature = "iroh")]
            TransportKind::Iroh(transport) => transport.bind().await.map(Listener::Iroh),
            #[cfg(feature = "websocket")]
            TransportKind::WebSocket(transport) => transport.bind().await.map(Listener::WebSocket),
        }
    }
}

#[cfg(feature = "iroh")]
impl From<IrohTransport> for Transport {
    fn from(transport: IrohTransport) -> Self {
        Self(TransportKind::Iroh(transport))
    }
}

#[cfg(feature = "websocket")]
impl From<WebSocketTransport> for Transport {
    fn from(transport: WebSocketTransport) -> Self {
        Self(TransportKind::WebSocket(transport))
    }
}

/// A bound [`Transport`], which accepts clients once served.
pub(crate) enum Listener {
    #[cfg(feature = "iroh")]
    Iroh(IrohListener),
    #[cfg(feature = "websocket")]
    WebSocket(WebSocketListener),
}

impl Listener {
    /// Serve until `setup`'s shutdown is cancelled, or the returned future is dropped.
    pub(crate) async fn serve<B: BackendIr>(
        self,
        setup: SessionSetup<B>,
    ) -> Result<(), ServeError> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(listener) => listener.serve(setup).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(listener) => listener.serve(setup).await,
        }
    }
}
