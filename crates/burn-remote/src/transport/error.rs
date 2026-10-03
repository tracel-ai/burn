//! Why opening a stream to a peer failed, sorted by whether trying again can succeed.

use core::fmt;

#[cfg(feature = "iroh")]
use iroh::{
    address_lookup::AddressLookupFailed,
    endpoint::{ConnectError, ConnectWithOptsError, ConnectingError, ConnectionError},
};

/// A failed attempt at opening a stream to a peer. It does not name the peer: the caller does.
#[derive(Debug)]
pub(crate) enum OpenError {
    /// The peer is not there yet: its server has not opened its port or published its address.
    /// Holds the reason, as a clause for a longer message.
    NotReachableYet(&'static str),
    /// No address was given and the endpoint cannot look one up.
    #[cfg(feature = "iroh")]
    NoAddress,
    /// Anything waiting cannot fix.
    Failed(String),
}

#[cfg(feature = "iroh")]
impl From<ConnectError> for OpenError {
    fn from(err: ConnectError) -> Self {
        match err {
            ConnectError::Connect {
                source:
                    ConnectWithOptsError::NoAddress {
                        source: AddressLookupFailed::NoResults { .. },
                        ..
                    },
                ..
            } => Self::NotReachableYet("no address is known"),
            ConnectError::Connect {
                source:
                    ConnectWithOptsError::NoAddress {
                        source: AddressLookupFailed::NoServiceConfigured { .. },
                        ..
                    },
                ..
            } => Self::NoAddress,
            ConnectError::Connecting {
                source:
                    ConnectingError::ConnectionError {
                        source: ConnectionError::TimedOut,
                        ..
                    },
                ..
            } => Self::Failed(
                "timed out: nothing answered, directly or through a relay. A direct path needs \
                 the server's UDP port reachable through any firewall, and a relay needs both \
                 sides able to reach it"
                    .into(),
            ),
            err => Self::Failed(err.to_string()),
        }
    }
}

#[cfg(all(feature = "client", feature = "websocket"))]
impl From<burn_communication::websocket::WsClientError> for OpenError {
    fn from(err: burn_communication::websocket::WsClientError) -> Self {
        if err.is_connection_refused() {
            Self::NotReachableYet("connection refused")
        } else {
            Self::Failed(err.to_string())
        }
    }
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotReachableYet(reason) => write!(f, "not reachable yet ({reason})"),
            #[cfg(feature = "iroh")]
            Self::NoAddress => {
                f.write_str("no address was given and the endpoint cannot look one up")
            }
            Self::Failed(message) => f.write_str(message),
        }
    }
}
