use burn_std::future::DynFut;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::str::FromStr;

/// Allows nodes to find each other
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Address {
    pub(crate) inner: String,
}

impl FromStr for Address {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            inner: s.to_string(),
        })
    }
}

impl Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
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
    /// Returns None if the connection can't be done.
    fn connect(address: Address, route: &str) -> DynFut<Option<Self::Channel>>;
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
