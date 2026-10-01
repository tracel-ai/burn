use core::fmt;

use crate::shared::{PROTOCOL_VERSION, SessionRefusal};

/// Why a remote device could not be connected.
///
/// [`Unreachable`](Self::Unreachable) and [`Handshake`](Self::Handshake) can pass on a later try;
/// the others need a change to the client or the server.
#[derive(Debug)]
pub enum ConnectError {
    /// No address was given and the endpoint has no way to look one up: relays are disabled, or
    /// the endpoint was bound without an address lookup.
    #[cfg(feature = "iroh")]
    NoAddress,
    /// The Iroh endpoint Burn binds could not be bound.
    #[cfg(feature = "iroh")]
    Bind {
        /// Iroh's reason.
        source: iroh::endpoint::BindError,
    },
    /// The host's settings cannot work together, such as an application endpoint combined with
    /// relays for an endpoint Burn binds.
    InvalidConfiguration {
        /// Which settings conflict.
        reason: String,
    },
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
    /// Burn release. The higher version is the newer release.
    IncompatibleProtocol {
        /// The version this client speaks.
        client_version: u16,
        /// The version the server speaks.
        server_version: u16,
    },
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
            // At the client's own version, the server could not read the handshake itself.
            SessionRefusal::IncompatibleProtocol { server_version }
                if server_version == PROTOCOL_VERSION =>
            {
                Self::Handshake {
                    reason: "the server could not read this client's handshake".into(),
                }
            }
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
            #[cfg(feature = "iroh")]
            Self::Bind { source } => write!(f, "cannot bind an Iroh endpoint: {source}"),
            Self::InvalidConfiguration { reason } => write!(f, "invalid remote host: {reason}"),
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
            #[cfg(feature = "iroh")]
            Self::Bind { source } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_protocol_refusal_at_the_clients_own_version_is_a_handshake_error() {
        let error = ConnectError::from(SessionRefusal::IncompatibleProtocol {
            server_version: PROTOCOL_VERSION,
        });
        assert!(matches!(error, ConnectError::Handshake { .. }), "{error:?}");
    }

    #[test]
    fn a_protocol_refusal_from_another_version_names_both_versions() {
        let server_version = PROTOCOL_VERSION + 1;
        let error = ConnectError::from(SessionRefusal::IncompatibleProtocol { server_version });
        assert!(
            matches!(
                error,
                ConnectError::IncompatibleProtocol {
                    client_version: PROTOCOL_VERSION,
                    server_version: refused,
                } if refused == server_version
            ),
            "{error:?}"
        );
    }
}
