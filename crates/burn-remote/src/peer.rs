use core::fmt;

#[cfg(feature = "websocket")]
use burn_communication::Address;
use serde::{Deserialize, Serialize};

/// Stable identity of a Burn Remote peer.
///
/// Network paths are deliberately not part of the identity. An Iroh peer keeps the same
/// identity while moving between direct addresses and relays.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum PeerId {
    /// An Iroh endpoint, authenticated by its public key.
    #[cfg(feature = "iroh")]
    Iroh(iroh::EndpointId),
    /// A legacy WebSocket endpoint.
    #[cfg(feature = "websocket")]
    WebSocket(Address),
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(id) => write!(f, "iroh://{id}"),
            #[cfg(feature = "websocket")]
            Self::WebSocket(address) => address.fmt(f),
        }
    }
}

// Only the Iroh server resolves a `PeerId` back to an endpoint id (to address tensor transfers).
#[cfg(all(feature = "iroh", feature = "server"))]
impl PeerId {
    /// The Iroh endpoint id, or `None` for a non-Iroh peer.
    pub(crate) fn into_iroh_id(self) -> Option<iroh::EndpointId> {
        #[cfg(feature = "websocket")]
        return match self {
            Self::Iroh(id) => Some(id),
            Self::WebSocket(_) => None,
        };
        #[cfg(not(feature = "websocket"))]
        {
            let Self::Iroh(id) = self;
            Some(id)
        }
    }
}

/// A peer identity plus the mutable network paths that may be used to reach it.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum PeerAddr {
    /// An Iroh endpoint address. It may contain direct and relay paths, or only an endpoint id
    /// when the configured Iroh address lookup can resolve it.
    #[cfg(feature = "iroh")]
    Iroh(iroh::EndpointAddr),
    /// A legacy WebSocket address.
    #[cfg(feature = "websocket")]
    WebSocket(Address),
}

/// A compute server's stable identity: the secret stays on the server, and the public
/// [`id`](Self::id) it yields is the address clients dial. Generate one with [`random`](Self::random)
/// and persist [`to_bytes`](Self::to_bytes) for a stable address across restarts, or derive it from a
/// seed with [`from_bytes`](Self::from_bytes).
#[cfg(feature = "iroh")]
#[derive(Clone)]
pub struct RemoteSecret(iroh::SecretKey);

#[cfg(feature = "iroh")]
impl RemoteSecret {
    /// A fresh random identity. Persist [`to_bytes`](Self::to_bytes) to reuse the same address later.
    pub fn random() -> Self {
        Self(iroh::SecretKey::generate())
    }

    /// A deterministic identity from 32 seed bytes (e.g. a hash of an application name).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(iroh::SecretKey::from_bytes(&bytes))
    }

    /// The raw 32 bytes, to persist and reload a stable identity.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    /// The public identity clients dial.
    pub fn id(&self) -> iroh::EndpointId {
        self.0.public()
    }

    pub(crate) fn secret_key(&self) -> iroh::SecretKey {
        self.0.clone()
    }
}

impl PeerAddr {
    /// Return the stable peer identity, excluding dialing hints.
    pub fn id(&self) -> PeerId {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(address) => PeerId::Iroh(address.id),
            #[cfg(feature = "websocket")]
            Self::WebSocket(address) => PeerId::WebSocket(address.clone()),
        }
    }

    /// Return true when this is an Iroh peer.
    pub fn is_iroh(&self) -> bool {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(_) => true,
            #[cfg(feature = "websocket")]
            Self::WebSocket(_) => false,
        }
    }
}

impl fmt::Display for PeerAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id().fmt(f)
    }
}

#[cfg(feature = "iroh")]
impl From<iroh::EndpointAddr> for PeerAddr {
    fn from(value: iroh::EndpointAddr) -> Self {
        Self::Iroh(value)
    }
}

#[cfg(feature = "iroh")]
impl From<iroh::EndpointId> for PeerAddr {
    fn from(value: iroh::EndpointId) -> Self {
        Self::Iroh(value.into())
    }
}

#[cfg(feature = "websocket")]
impl From<Address> for PeerAddr {
    fn from(value: Address) -> Self {
        Self::WebSocket(value)
    }
}
