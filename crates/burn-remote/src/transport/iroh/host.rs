use core::hash::{Hash, Hasher};
#[cfg(not(target_family = "wasm"))]
use std::net::SocketAddr;

use iroh::{Endpoint, EndpointAddr, EndpointId};

use super::{node::RemoteNode, relays::IrohRelays};
use crate::ConnectError;

/// An Iroh compute server as a client reaches it: its id and any addresses to try directly, and
/// either the relays of an endpoint Burn binds or the application's own endpoint.
///
/// Building one binds nothing. Burn binds one endpoint per [`IrohRelays`] value the first time a
/// host needs it, and every host with that value shares it.
#[derive(Clone, Debug)]
pub struct IrohHost {
    addr: EndpointAddr,
    relays: Option<IrohRelays>,
    endpoint: Option<Endpoint>,
}

impl IrohHost {
    /// The server at `server`: its id alone, or an [`EndpointAddr`] that also lists addresses.
    /// Reached through n0's public relays, from an endpoint Burn binds, unless set otherwise.
    pub fn new(server: impl Into<EndpointAddr>) -> Self {
        Self {
            addr: server.into(),
            relays: None,
            endpoint: None,
        }
    }

    /// The relays the server uses, applied to the endpoint Burn binds.
    pub fn with_relays(mut self, relays: IrohRelays) -> Self {
        self.relays = Some(relays);
        self
    }

    /// An address to try directly, required when relays are disabled. Give every address a host
    /// name resolves to: the server may listen on only one of IPv4 and IPv6.
    #[cfg(not(target_family = "wasm"))]
    pub fn with_address(mut self, address: SocketAddr) -> Self {
        self.addr = self.addr.with_ip_addr(address);
        self
    }

    /// Dial from `endpoint`, shared with the application's other Iroh protocols, instead of an
    /// endpoint Burn binds. Its own relay settings then apply, so this excludes
    /// [`with_relays`](Self::with_relays).
    ///
    /// Every blocking call on the device, a read included, waits on the runtime that bound
    /// `endpoint`, so bind it on a multi-thread runtime rather than a current-thread one.
    pub fn with_endpoint(mut self, endpoint: Endpoint) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    pub(crate) fn app_endpoint(&self) -> Option<EndpointId> {
        self.endpoint.as_ref().map(Endpoint::id)
    }

    fn relays(&self) -> IrohRelays {
        self.relays.clone().unwrap_or_default()
    }

    /// Refuse settings that cannot reach the server before anything is bound.
    pub(crate) fn validate(&self) -> Result<(), ConnectError> {
        if self.endpoint.is_some() && self.relays.is_some() {
            return Err(ConnectError::InvalidConfiguration {
                reason: "an IrohHost dials from the application's endpoint or from one Burn binds \
                         with the given relays, not both"
                    .into(),
            });
        }
        if self.endpoint.is_some() {
            return Ok(());
        }
        let relays = self.relays();
        #[cfg(target_family = "wasm")]
        if relays == IrohRelays::Disabled {
            return Err(ConnectError::InvalidConfiguration {
                reason: "a browser reaches Iroh servers only through relays".into(),
            });
        }
        if relays == IrohRelays::Disabled && self.addr.ip_addrs().next().is_none() {
            return Err(ConnectError::NoAddress);
        }
        Ok(())
    }

    /// The server's address with the private relay it uses, if any.
    pub(crate) fn dial_addr(&self) -> EndpointAddr {
        match &self.relays {
            Some(IrohRelays::Private { url }) => self.addr.clone().with_relay_url(url.clone()),
            _ => self.addr.clone(),
        }
    }

    /// The node to dial from: the one shared by every user of the application's endpoint, or the
    /// one Burn binds for these relays.
    pub(crate) async fn node(&self) -> Result<RemoteNode, ConnectError> {
        match &self.endpoint {
            Some(endpoint) => RemoteNode::for_endpoint(endpoint)
                .map_err(|reason| ConnectError::InvalidConfiguration { reason }),
            None => RemoteNode::for_relays(&self.relays())
                .await
                .map_err(|source| ConnectError::Bind {
                    source: source.into(),
                }),
        }
    }
}

impl PartialEq for IrohHost {
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr
            && self.relays == other.relays
            && self.app_endpoint() == other.app_endpoint()
    }
}

impl Eq for IrohHost {}

impl Hash for IrohHost {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.addr.hash(state);
        self.relays.hash(state);
        self.app_endpoint().hash(state);
    }
}

impl From<EndpointAddr> for IrohHost {
    fn from(server: EndpointAddr) -> Self {
        Self::new(server)
    }
}

impl From<EndpointId> for IrohHost {
    fn from(server: EndpointId) -> Self {
        Self::new(server)
    }
}
