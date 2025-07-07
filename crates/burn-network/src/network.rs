use burn_common::future::DynFut;
use std::fmt::Debug;

pub trait NetworkClient: Send + 'static {
    type ClientStream: NetworkStream;

    fn connect(address: String) -> DynFut<Option<Self::ClientStream>>;
}

pub struct NetworkMessage {
    pub data: bytes::Bytes,
}

pub trait NetworkServer: Send + Sync + 'static {
    type State: Clone + Send + Sync + 'static;
    type ServerStream: NetworkStream;

    fn new(port: u16) -> Self;

    fn route<C, Fut>(self, path: &str, callback: C) -> Self
    where
        C: FnOnce(Self::State, Self::ServerStream) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static;

    fn serve<F>(
        self,
        state: Self::State,
        shutdown: F,
    ) -> impl std::future::Future<Output = ()> + Send
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
