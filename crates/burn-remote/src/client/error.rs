use core::fmt;

/// Why a remote session could not be opened.
#[derive(Debug)]
pub(crate) enum SessionError {
    /// No session stream could be opened to the server.
    Unreachable { reason: String },
    /// The server answered the handshake with why it will not serve the session.
    Refused { reason: String },
    /// The server was reached, but the handshake broke off or its reply made no sense.
    Handshake { reason: String },
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreachable { reason } => f.write_str(reason),
            Self::Refused { reason } => write!(f, "the server refused the session: {reason}"),
            Self::Handshake { reason } => write!(f, "the session handshake failed: {reason}"),
        }
    }
}
