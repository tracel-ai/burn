use core::fmt;

use crate::shared::{PROTOCOL_VERSION, SessionRefusal};

/// Why a remote device could not be connected.
///
/// [`Unreachable`](Self::Unreachable) and [`Handshake`](Self::Handshake) can pass on a later try;
/// the others need a change to the client or the server.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConnectError {
    /// No address was given and the endpoint has no way to look one up: relays are disabled, or
    /// the endpoint was bound without an address lookup.
    #[cfg(feature = "iroh")]
    NoAddress,
    /// The local Iroh endpoint could not be bound.
    #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
    #[non_exhaustive]
    Bind {
        /// Iroh's reason.
        source: iroh::endpoint::BindError,
    },
    /// The runtime shut down before the connection was attempted.
    #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
    Interrupted,
    /// No session could be opened with the server: it is not running, nothing answered at its
    /// addresses, or another endpoint answered there.
    #[non_exhaustive]
    Unreachable {
        /// What failed.
        reason: String,
    },
    /// The server's authorizer rejected the credential. Its reason is in the server's log.
    Unauthorized,
    /// The server does not host the device index asked for.
    #[non_exhaustive]
    NoSuchDevice {
        /// How many devices the server hosts.
        device_count: usize,
    },
    /// The server speaks another version of the Burn Remote protocol: build both with the same
    /// Burn release. The higher version is the newer release.
    #[non_exhaustive]
    IncompatibleProtocol {
        /// The version this client speaks.
        client_version: u16,
        /// The version the server speaks.
        server_version: u16,
    },
    /// The server was reached, but the session handshake broke off or its reply made no sense. A
    /// server on an older Burn release refuses a session by closing it unanswered, which lands here.
    #[non_exhaustive]
    Handshake {
        /// What went wrong.
        reason: String,
    },
}

impl From<SessionRefusal> for ConnectError {
    fn from(refusal: SessionRefusal) -> Self {
        match refusal {
            SessionRefusal::Unauthorized => Self::Unauthorized,
            SessionRefusal::NoSuchDevice { device_count } => Self::NoSuchDevice {
                device_count: device_count as usize,
            },
            SessionRefusal::IncompatibleProtocol { server_version } => Self::IncompatibleProtocol {
                client_version: PROTOCOL_VERSION,
                server_version,
            },
        }
    }
}

impl fmt::Display for ConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "iroh")]
            Self::NoAddress => {
                f.write_str("no address was given and the endpoint cannot look one up")
            }
            #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
            Self::Bind { source } => write!(f, "cannot bind an Iroh endpoint: {source}"),
            #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
            Self::Interrupted => f.write_str("the runtime shut down before connecting"),
            Self::Unreachable { reason } => write!(f, "the server cannot be reached: {reason}"),
            Self::Unauthorized => f.write_str("the server's authorizer rejected the credential"),
            Self::NoSuchDevice { device_count } => {
                write!(f, "the server hosts only {device_count} device(s)")
            }
            Self::IncompatibleProtocol {
                client_version,
                server_version,
            } => write!(
                f,
                "the server speaks version {server_version} of the Burn Remote protocol, and this \
                 client version {client_version}"
            ),
            Self::Handshake { reason } => write!(f, "the session handshake failed: {reason}"),
        }
    }
}

impl std::error::Error for ConnectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
            Self::Bind { source } => Some(source),
            _ => None,
        }
    }
}
