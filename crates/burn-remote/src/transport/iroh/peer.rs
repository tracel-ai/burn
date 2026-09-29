use std::net::SocketAddr;

use iroh::{EndpointAddr, EndpointId, TransportAddr, endpoint::BindError};

use super::relays::IrohRelays;

/// An Iroh compute server as a client dials it: its id, the relays it uses, the addresses to try
/// directly, and the credential its authorizer checks.
#[derive(Clone, Debug)]
pub struct IrohPeer {
    id: EndpointId,
    relays: IrohRelays,
    addresses: Vec<SocketAddr>,
    credential: Vec<u8>,
    segmentation_offload: bool,
}

impl IrohPeer {
    /// The server whose id is `id`, reached through n0's public relays, with no credential.
    pub fn new(id: EndpointId) -> Self {
        Self {
            id,
            relays: IrohRelays::default(),
            addresses: Vec::new(),
            credential: Vec::new(),
            segmentation_offload: false,
        }
    }

    /// The relays the server uses.
    pub fn relays(mut self, relays: IrohRelays) -> Self {
        self.relays = relays;
        self
    }

    /// An address to try directly, required when relays are disabled. Give every address a host
    /// name resolves to: the server may listen on only one of IPv4 and IPv6.
    pub fn address(mut self, address: SocketAddr) -> Self {
        self.addresses.push(address);
        self
    }

    /// What the server's authorizer checks, such as the token of a
    /// [`TokenAuthorizer`](crate::server::TokenAuthorizer).
    pub fn credential(mut self, credential: impl Into<Vec<u8>>) -> Self {
        self.credential = credential.into();
        self
    }

    /// Send segmentation-offloaded (GSO) batches, which costs less CPU per byte on fast links.
    /// Off by default: iroh keeps sending them after the kernel refuses one, which kills every
    /// open connection ([iroh#4555](https://github.com/n0-computer/iroh/issues/4555)). Turn it on
    /// only where the network stack is known to accept them.
    pub fn segmentation_offload(mut self, enabled: bool) -> Self {
        self.segmentation_offload = enabled;
        self
    }

    pub(crate) async fn bind(&self) -> Result<iroh::Endpoint, BindError> {
        self.relays
            .endpoint_builder(self.segmentation_offload)
            .bind()
            .await
    }

    pub(crate) fn addr(&self) -> EndpointAddr {
        let addr = EndpointAddr::new(self.id)
            .with_addrs(self.addresses.iter().copied().map(TransportAddr::Ip));
        match &self.relays {
            IrohRelays::Private(url) => addr.with_relay_url(url.clone()),
            IrohRelays::Public | IrohRelays::Disabled => addr,
        }
    }

    pub(crate) fn credential_bytes(&self) -> &[u8] {
        &self.credential
    }
}
