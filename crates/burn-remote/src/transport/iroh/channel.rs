use std::sync::Arc;

use super::{
    protocol::{AllowAll, PeerAuthorizer},
    relays::IrohRelays,
    secret::RemoteSecret,
};

/// How a server serves over Iroh: its identity, its relays, the port it binds, and whom it serves.
#[derive(Clone)]
pub struct IrohChannel {
    pub(crate) secret: RemoteSecret,
    pub(crate) relays: IrohRelays,
    pub(crate) port: Option<u16>,
    pub(crate) authorizer: Arc<dyn PeerAuthorizer>,
    pub(crate) segmentation_offload: bool,
}

impl IrohChannel {
    /// Serve as `secret`'s id through n0's public relays, on a port the OS picks, to every peer.
    pub fn new(secret: RemoteSecret) -> Self {
        Self {
            secret,
            relays: IrohRelays::default(),
            port: None,
            authorizer: Arc::new(AllowAll),
            segmentation_offload: false,
        }
    }

    /// The relays to reach clients through.
    pub fn relays(mut self, relays: IrohRelays) -> Self {
        self.relays = relays;
        self
    }

    /// Bind UDP `port` on every IPv4 and IPv6 interface, so clients can dial it directly. Needed
    /// when relays are disabled, since nothing else tells clients where the server is.
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Serve only the sessions `authorizer` accepts.
    pub fn authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.authorizer = Arc::new(authorizer);
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

    /// The id clients dial.
    pub fn id(&self) -> iroh::EndpointId {
        self.secret.id()
    }
}

impl core::fmt::Debug for IrohChannel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The public identity only, never the secret key material.
        f.debug_struct("IrohChannel")
            .field("id", &self.id())
            .field("relays", &self.relays)
            .field("port", &self.port)
            .field("segmentation_offload", &self.segmentation_offload)
            .finish_non_exhaustive()
    }
}
