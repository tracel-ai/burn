use core::fmt;

use crate::shared::SessionRefusal;

/// Why a remote session could not be opened.
#[derive(Debug)]
pub(crate) enum SessionOpenError {
    /// No session stream could be opened to the server.
    Unreachable { reason: String },
    /// The session cannot be served: the server said why, or the client found the server's
    /// protocol version is not its own.
    Refused { refusal: SessionRefusal },
    /// The server was reached, but the handshake broke off or its reply made no sense.
    Handshake { reason: String },
}

impl fmt::Display for SessionOpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreachable { reason } => write!(f, "the server cannot be reached: {reason}"),
            Self::Refused { refusal } => refusal.fmt(f),
            Self::Handshake { reason } => write!(f, "the session handshake failed: {reason}"),
        }
    }
}
