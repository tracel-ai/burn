//! Who a server serves.

use core::fmt;
use std::net::SocketAddr;

#[cfg(feature = "iroh")]
use iroh::EndpointId;

use crate::Credential;

/// The client asking for a session, as its transport identifies it.
///
/// `WebSocket` is there in every build, so a match on this keeps compiling when another crate in
/// the build enables the WebSocket transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClientId {
    /// The client's Iroh endpoint id, which the transport authenticates.
    #[cfg(feature = "iroh")]
    Iroh(EndpointId),
    /// The address the client connected from. Unauthenticated: anything on the network path can
    /// claim it.
    WebSocket(SocketAddr),
}

impl fmt::Display for ClientId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            #[cfg(feature = "iroh")]
            Self::Iroh(id) => write!(f, "Iroh client {id}"),
            Self::WebSocket(address) => write!(f, "WebSocket client {address}"),
        }
    }
}

/// A session a client asks to open, presented to the server's [`PeerAuthorizer`].
#[derive(Debug)]
pub struct AuthorizationRequest<'a> {
    /// Who is asking.
    pub client: ClientId,
    /// Which of the server's devices, by its position in the list the server hosts.
    pub device_index: u32,
    /// What the client presented, empty when it set none.
    pub credential: &'a Credential,
}

/// Which sessions a server opens. A closure taking an [`AuthorizationRequest`] is one.
///
/// Called on every session a client opens, before the server tells it anything about its devices.
/// Tensor transfers between servers do not pass through it: they carry a capability the client's
/// session granted.
pub trait PeerAuthorizer: Send + Sync + 'static {
    /// `Ok(())` to open the session, or why not, for the server's log: the client is told only that
    /// it was refused.
    fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<(), String>;
}

impl<F> PeerAuthorizer for F
where
    F: Fn(AuthorizationRequest<'_>) -> Result<(), String> + Send + Sync + 'static,
{
    fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<(), String> {
        self(request)
    }
}

/// Opens every session, which is what a server does unless given another authorizer. Fit for a
/// server only its own machine or a trusted network can reach.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllowAll;

impl PeerAuthorizer for AllowAll {
    fn authorize(&self, _request: AuthorizationRequest<'_>) -> Result<(), String> {
        Ok(())
    }
}

/// Opens only the sessions of clients that present this token, which a client sets with
/// `RemoteHost::with_credential`.
///
/// Over WebSocket the token travels unencrypted: it stops stray clients on a trusted network, not
/// someone reading the traffic.
#[derive(Clone)]
pub struct TokenAuthorizer {
    // A digest compares in constant time whatever the credential's length, so timing reveals
    // neither the token nor its length.
    digest: blake3::Hash,
}

impl TokenAuthorizer {
    /// Serve the clients that present `token`.
    ///
    /// # Errors
    ///
    /// An empty token, which every client that sets no credential presents.
    pub fn new(token: impl Into<Credential>) -> Result<Self, EmptyToken> {
        let token = token.into();
        if token.is_empty() {
            return Err(EmptyToken);
        }
        Ok(Self {
            digest: blake3::hash(token.as_bytes()),
        })
    }
}

impl PeerAuthorizer for TokenAuthorizer {
    fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<(), String> {
        if blake3::hash(request.credential.as_bytes()) == self.digest {
            Ok(())
        } else {
            Err(format!("{} presented the wrong token", request.client))
        }
    }
}

impl fmt::Debug for TokenAuthorizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenAuthorizer").finish_non_exhaustive()
    }
}

/// A [`TokenAuthorizer`] was given an empty token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmptyToken;

impl fmt::Display for EmptyToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a token authorizer needs a non-empty token")
    }
}

impl std::error::Error for EmptyToken {}

#[cfg(all(test, feature = "iroh"))]
mod tests {
    use super::*;

    #[test]
    fn an_empty_token_is_refused() {
        assert_eq!(TokenAuthorizer::new("").unwrap_err(), EmptyToken);
    }

    #[test]
    fn only_the_token_is_let_in() {
        let authorizer = TokenAuthorizer::new("secret-token").unwrap();
        assert!(
            authorizer
                .authorize(request(&"secret-token".into()))
                .is_ok()
        );
        assert!(
            authorizer
                .authorize(request(&"secret-toke".into()))
                .is_err()
        );
        assert!(
            authorizer
                .authorize(request(&Credential::default()))
                .is_err()
        );
    }

    fn request(credential: &Credential) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            client: ClientId::Iroh(iroh::SecretKey::generate().public()),
            device_index: 0,
            credential,
        }
    }
}
