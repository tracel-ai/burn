use crate::{
    base::{Address, CommunicationChannel, CommunicationError, Message, ProtocolClient},
    websocket::base::parse_ws_address,
};
use burn_std::future::DynFut;
use futures::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{self, protocol::WebSocketConfig},
};

#[derive(Clone)]
pub struct WsClient;

impl ProtocolClient for WsClient {
    type Channel = WsClientChannel;
    type Error = WsClientError;

    fn connect(address: Address, route: &str) -> DynFut<Option<WsClientChannel>> {
        Box::pin(connect_ws(address, route.to_owned()))
    }
}

/// Open a new WebSocket connection to the address
async fn connect_ws(address: Address, route: String) -> Option<WsClientChannel> {
    let address = parse_ws_address(address).ok()?;
    let address = format!("{address}/{route}");
    const MB: usize = 1024 * 1024;
    let (stream, _) = connect_async_with_config(
        address.clone(),
        Some(
            WebSocketConfig::default()
                .write_buffer_size(0)
                .max_message_size(None)
                .max_frame_size(Some(MB * 512))
                .accept_unmasked_frames(true)
                .read_buffer_size(64 * 1024), // 64 KiB (previous default)
        ),
        true,
    )
    .await
    .ok()?;

    Some(WsClientChannel { inner: stream })
}
pub struct WsClientChannel {
    inner: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

impl CommunicationChannel for WsClientChannel {
    type Error = WsClientError;

    async fn send(&mut self, msg: Message) -> Result<(), WsClientError> {
        self.inner
            .send(tungstenite::Message::Binary(msg.data))
            .await?;

        Ok(())
    }

    async fn recv(&mut self) -> Result<Option<Message>, WsClientError> {
        match self.inner.next().await {
            Some(next) => match next {
                Ok(tungstenite::Message::Binary(data)) => Ok(Some(Message { data })),
                Ok(tungstenite::Message::Close(_close_frame)) => Ok(None),
                Err(err) => Err(WsClientError::Tungstenite(err)),
                msg => Err(WsClientError::UnknownMessage(format!("{msg:?}"))),
            },
            None => todo!(),
        }
    }

    async fn close(&mut self) -> Result<(), WsClientError> {
        let reason = "Peer is closing".to_string();

        self.inner
            .send(tungstenite::Message::Close(Some(
                tungstenite::protocol::CloseFrame {
                    code: tungstenite::protocol::frame::coding::CloseCode::Normal,
                    reason: reason.clone().into(),
                },
            )))
            .await?;

        Ok(())
    }
}

#[derive(Debug)]
pub enum WsClientError {
    Io(std::io::Error),
    Tungstenite(tungstenite::Error),
    UnknownMessage(String),
    Other(String),
}
impl CommunicationError for WsClientError {}

impl From<std::io::Error> for WsClientError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<tungstenite::Error> for WsClientError {
    fn from(err: tungstenite::Error) -> Self {
        Self::Tungstenite(err)
    }
}
