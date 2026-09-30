use core::fmt;

use crate::shared::SessionRefusal;

/// Why a remote device could not be connected.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConnectError {
    /// Relays are disabled and no address was given, so an endpoint Burn binds cannot find the
    /// server.
    #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
    NoAddress,
    /// The local Iroh endpoint could not be bound.
    #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
    Bind {
        /// Iroh's reason.
        source: iroh::endpoint::BindError,
    },
    /// The runtime shut down before the connection was attempted.
    #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
    Interrupted,
    /// No session could be opened with the server: it is not running, nothing answered at its
    /// addresses, or another endpoint answered there.
    Unreachable {
        /// What failed.
        reason: String,
    },
    /// The server's authorizer rejected the credential. Its reason is in the server's log.
    Unauthorized,
    /// The server does not host the device index asked for.
    NoSuchDevice {
        /// How many devices the server hosts.
        device_count: usize,
    },
    /// The server speaks another version of the Burn Remote protocol: build both with the same
    /// Burn release.
    IncompatibleProtocol,
    /// The server was reached, but the session handshake broke off or its reply made no sense. A
    /// server on an older Burn release refuses a session by closing it unanswered, which lands here.
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
            SessionRefusal::IncompatibleProtocol => Self::IncompatibleProtocol,
        }
    }
}

impl fmt::Display for ConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
            Self::NoAddress => f.write_str("relays are disabled and no address was given"),
            #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
            Self::Bind { source } => write!(f, "cannot bind an Iroh endpoint: {source}"),
            #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
            Self::Interrupted => f.write_str("the runtime shut down before connecting"),
            Self::Unreachable { reason } => write!(f, "the server cannot be reached: {reason}"),
            Self::Unauthorized => f.write_str("the server's authorizer rejected the credential"),
            Self::NoSuchDevice { device_count } => {
                write!(f, "the server hosts only {device_count} device(s)")
            }
            Self::IncompatibleProtocol => {
                f.write_str("the server speaks another version of the Burn Remote protocol")
            }
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
