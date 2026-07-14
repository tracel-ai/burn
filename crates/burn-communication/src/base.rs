use burn_std::future::DynFut;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::str::FromStr;

/// A parsed network endpoint used by nodes to find each other.
///
/// Construction normalizes the input so that equivalent spellings of the same endpoint
/// compare equal: a missing scheme defaults to `ws`, and a trailing path/slash is dropped.
/// As a result `"localhost:3000"`, `"ws://localhost:3000"`, and `"ws://localhost:3000/"`
/// all parse to the same `Address`. Equality is over `(scheme, host, port)`, so two devices
/// that point at the same server (e.g. different device indices) share one `Address` — that
/// is the signal the remote backend uses to detect a same-host transfer.
///
/// Note: host names are compared textually, not resolved — `localhost` and `127.0.0.1` are
/// *not* considered equal. Aliases simply fall back to the network transfer path.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Address {
    scheme: String,
    host: String,
    port: Option<u16>,
}

impl Address {
    /// The scheme component (e.g. `ws`), lowercased. Defaults to `ws` when none was given.
    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    /// The host component (host name or IP literal), exactly as written.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// The port component, if one was specified.
    pub fn port(&self) -> Option<u16> {
        self.port
    }

    /// Parse an endpoint string into its components, applying canonicalization.
    fn parse(s: &str) -> Self {
        let s = s.trim();
        let (scheme, rest) = match s.split_once("://") {
            Some((scheme, rest)) => (scheme.to_ascii_lowercase(), rest),
            None => ("ws".to_string(), s),
        };

        // Drop any path/query/fragment — only the authority identifies the endpoint.
        let authority = rest
            .split(['/', '?', '#'])
            .next()
            .unwrap_or(rest)
            .trim_end_matches('/');

        // Split host:port from the right so IPv6 literals like `[::1]:3000` keep their
        // colons; only the final `:port` (when it parses as a port) is treated as the port.
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) if !host.is_empty() => match port.parse::<u16>() {
                Ok(port) => (host.to_string(), Some(port)),
                Err(_) => (authority.to_string(), None),
            },
            _ => (authority.to_string(), None),
        };

        Self { scheme, host, port }
    }
}

impl From<&str> for Address {
    fn from(s: &str) -> Self {
        Address::parse(s)
    }
}

impl FromStr for Address {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Address::parse(s))
    }
}

impl Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}://{}", self.scheme, self.host)?;
        if let Some(port) = self.port {
            write!(f, ":{port}")?;
        }
        Ok(())
    }
}

/// The protocol used for the communications.
pub trait Protocol: Clone + Send + Sync + 'static {
    /// The client implementation for the current protocol.
    type Client: ProtocolClient;
    /// The server implementation for the current protocol.
    type Server: ProtocolServer;
}

/// Error that happens during a communication.
pub trait CommunicationError: Debug + Send + 'static {}

/// The client is only used to create a [channel](CommunicationChannel), which should be use to
/// transmit information with the [server](ProtocolServer).
pub trait ProtocolClient: Send + Sync + 'static {
    /// Channel used by this protocol.
    type Channel: CommunicationChannel<Error = Self::Error>;
    /// The error type.
    type Error: CommunicationError;

    /// Opens a new [channel](CommunicationChannel) with the current protocol at the given
    /// [address](Address) and route.
    ///
    /// * `address` - Address to connect to
    /// * `route` - The name of the route (no slashes)
    ///
    /// Returns `Err(Self::Error)` if the connection couldn't be established (address parse
    /// failure, server unreachable, handshake failure, …). The error carries enough context
    /// to identify the cause; callers should surface it rather than swallowing it.
    fn connect(address: Address, route: &str) -> DynFut<Result<Self::Channel, Self::Error>>;
}

/// Data sent and received by the client and server.
#[derive(new)]
pub struct Message {
    /// The data is always encoded as bytes.
    pub data: bytes::Bytes,
}

/// Defines how to create a server that respond to a [channel](CommunicationChannel).
pub trait ProtocolServer: Sized + Send + Sync + 'static {
    /// Channel used by this protocol.
    type Channel: CommunicationChannel<Error = Self::Error>;
    /// The error type.
    type Error: CommunicationError;

    /// Defines an endpoint with the function that responds.
    /// TODO Docs: does it need a slash?
    fn route<C, Fut>(self, path: &str, callback: C) -> Self
    where
        C: FnOnce(Self::Channel) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static;

    /// Start the server.
    fn serve<F>(
        self,
        shutdown: F,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static
    where
        F: Future<Output = ()> + Send + 'static;
}

/// Handles communications.
pub trait CommunicationChannel: Send + 'static {
    type Error: CommunicationError;

    /// Send a [message](Message) on the channel.
    fn send(
        &mut self,
        message: Message,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;

    /// Receive a [message](Message) on the channel and returns a new [response message](Message).
    fn recv(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Option<Message>, Self::Error>> + Send;

    fn close(&mut self) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_defaults_scheme_and_canonicalizes() {
        let bare = Address::from("localhost:3000");
        let with_scheme = Address::from("ws://localhost:3000");
        let trailing = Address::from("ws://localhost:3000/");

        // Equivalent spellings canonicalize to the same value.
        assert_eq!(bare, with_scheme);
        assert_eq!(with_scheme, trailing);
        assert_eq!(bare.scheme(), "ws");
        assert_eq!(bare.host(), "localhost");
        assert_eq!(bare.port(), Some(3000));
        assert_eq!(with_scheme.to_string(), "ws://localhost:3000");
    }

    #[test]
    fn address_distinguishes_port_host_and_scheme() {
        assert_ne!(
            Address::from("ws://host:3000"),
            Address::from("ws://host:3001")
        );
        assert_ne!(Address::from("ws://a:3000"), Address::from("ws://b:3000"));
        assert_ne!(
            Address::from("ws://host:3000"),
            Address::from("wss://host:3000")
        );
        // Host names are textual, not resolved.
        assert_ne!(
            Address::from("ws://localhost:3000"),
            Address::from("ws://127.0.0.1:3000")
        );
    }

    #[test]
    fn address_handles_ipv6_and_missing_port() {
        let v6 = Address::from("ws://[::1]:3000");
        assert_eq!(v6.host(), "[::1]");
        assert_eq!(v6.port(), Some(3000));

        let no_port = Address::from("ws://example.com");
        assert_eq!(no_port.host(), "example.com");
        assert_eq!(no_port.port(), None);
        assert_eq!(no_port.to_string(), "ws://example.com");
    }
}
