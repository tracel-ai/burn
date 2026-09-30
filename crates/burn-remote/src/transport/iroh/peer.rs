use std::net::SocketAddr;

use iroh::{EndpointAddr, EndpointId, TransportAddr, endpoint::BindError};

use super::relays::IrohRelays;
use crate::RemoteDevice;

/// An Iroh compute server as a client dials it: its id, the relays it uses, the addresses to try
/// directly, and the credential its authorizer checks. Built with [`IrohPeerBuilder`].
#[derive(Clone, Debug)]
pub struct IrohPeer {
    id: EndpointId,
    relays: IrohRelays,
    addresses: Vec<SocketAddr>,
    credential: Vec<u8>,
}

impl IrohPeer {
    /// Device `device_index` of this server, dialed from an endpoint bound for it.
    ///
    /// # Errors
    ///
    /// The endpoint could not be bound.
    ///
    /// # Panics
    ///
    /// The server refused the session, or could not be reached.
    #[cfg(not(target_family = "wasm"))]
    pub async fn connect(&self, device_index: usize) -> Result<RemoteDevice, BindError> {
        let endpoint = self.relays.endpoint_builder().bind().await?;
        let device = RemoteDevice::iroh_authorized(
            &endpoint,
            self.addr(),
            device_index,
            self.credential.clone(),
        );
        // The handshake blocks until the server answers, so it runs off the async workers.
        let connecting = device.clone();
        if let Err(err) = tokio::task::spawn_blocking(move || connecting.connect()).await {
            std::panic::resume_unwind(err.into_panic());
        }
        Ok(device)
    }

    fn addr(&self) -> EndpointAddr {
        let addr = EndpointAddr::new(self.id)
            .with_addrs(self.addresses.iter().copied().map(TransportAddr::Ip));
        match &self.relays {
            IrohRelays::Private(url) => addr.with_relay_url(url.clone()),
            IrohRelays::Public | IrohRelays::Disabled => addr,
        }
    }
}

/// Builds an [`IrohPeer`]. Unless set otherwise, it is reached through n0's public relays with no
/// credential.
#[derive(Clone, Debug)]
pub struct IrohPeerBuilder {
    peer: IrohPeer,
}

impl IrohPeerBuilder {
    /// The server whose id is `id`.
    pub fn new(id: EndpointId) -> Self {
        Self {
            peer: IrohPeer {
                id,
                relays: IrohRelays::default(),
                addresses: Vec::new(),
                credential: Vec::new(),
            },
        }
    }

    /// The relays the server uses.
    pub fn relays(mut self, relays: IrohRelays) -> Self {
        self.peer.relays = relays;
        self
    }

    /// An address to try directly, required when relays are disabled. Give every address a host
    /// name resolves to: the server may listen on only one of IPv4 and IPv6.
    pub fn address(mut self, address: SocketAddr) -> Self {
        self.peer.addresses.push(address);
        self
    }

    /// What the server's authorizer checks, such as the token of a
    /// [`TokenAuthorizer`](crate::server::TokenAuthorizer).
    pub fn credential(mut self, credential: impl Into<Vec<u8>>) -> Self {
        self.peer.credential = credential.into();
        self
    }

    /// The peer.
    pub fn build(self) -> IrohPeer {
        self.peer
    }
}
