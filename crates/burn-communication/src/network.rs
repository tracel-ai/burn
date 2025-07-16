use burn_common::future::DynFut;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::str::FromStr;

/// Allows nodes to find each other
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct NetworkAddress {
    pub(crate) inner: String,
}

impl FromStr for NetworkAddress {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            inner: s.to_string(),
        })
    }
}

impl Display for NetworkAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

pub trait Network: Clone + Send + Sync + 'static {
    type Client: NetworkClient;
    type Server: NetworkServer;
}

pub trait NetworkError: Debug + Send + 'static {}

pub trait NetworkClient: Send + Sync + 'static {
    type Stream: NetworkStream<Error = Self::Error>;
    type Error: NetworkError;

    fn connect(address: NetworkAddress, route: &str) -> DynFut<Option<Self::Stream>>;
}

pub struct NetworkMessage {
    pub data: bytes::Bytes,
}

pub trait NetworkServer: Sized + Send + Sync + 'static {
    type Stream: NetworkStream<Error = Self::Error>;
    type Error: NetworkError;

    fn new(port: u16) -> Self;

    fn route<C, Fut>(self, path: &str, callback: C) -> Self
    where
        C: FnOnce(Self::Stream) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static;

    fn serve<F>(
        self,
        shutdown: F,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static
    where
        F: Future<Output = ()> + Send + 'static;
}

pub trait NetworkStream: Send + 'static {
    type Error: Debug + Send;

    fn send(
        &mut self,
        bytes: bytes::Bytes,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;

    fn recv(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Option<NetworkMessage>, Self::Error>> + Send;

    fn close(&mut self) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;
}
