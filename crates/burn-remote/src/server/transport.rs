use burn_ir::BackendIr;

#[cfg(feature = "iroh")]
use crate::transport::iroh::IrohTransport;
#[cfg(feature = "websocket")]
use crate::transport::websocket::WebSocketTransport;

use super::{ServeError, SessionSetup};

/// How a server accepts clients. Built from an [`IrohTransport`] or a [`WebSocketTransport`].
#[derive(Debug)]
pub enum Transport {
    /// Over Iroh: any network, authenticated and encrypted.
    #[cfg(feature = "iroh")]
    Iroh(IrohTransport),
    /// Over WebSocket: the simplest setup on a trusted network, unencrypted.
    #[cfg(feature = "websocket")]
    WebSocket(WebSocketTransport),
}

impl Transport {
    pub(crate) async fn serve<B: BackendIr>(
        self,
        setup: SessionSetup<B>,
    ) -> Result<(), ServeError> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(transport) => transport.serve(setup).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(transport) => transport.serve(setup).await,
        }
    }
}

#[cfg(feature = "iroh")]
impl From<IrohTransport> for Transport {
    fn from(transport: IrohTransport) -> Self {
        Self::Iroh(transport)
    }
}

#[cfg(feature = "websocket")]
impl From<WebSocketTransport> for Transport {
    fn from(transport: WebSocketTransport) -> Self {
        Self::WebSocket(transport)
    }
}
