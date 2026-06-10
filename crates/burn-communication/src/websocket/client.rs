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

    fn connect(address: Address, route: &str) -> DynFut<Result<WsClientChannel, WsClientError>> {
        Box::pin(connect_ws(address, route.to_owned()))
    }
}

/// Open a new WebSocket connection to the address.
async fn connect_ws(address: Address, route: String) -> Result<WsClientChannel, WsClientError> {
    let address = parse_ws_address(address).map_err(WsClientError::Address)?;
    let url = format!("{address}/{route}");
    const MB: usize = 1024 * 1024;
    let (stream, _) = connect_async_with_config(
        url,
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
    .await?;

    Ok(WsClientChannel { inner: stream })
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
        // Keep reading until we get a data frame, a close, an error, or the stream ends.
        // Ping/Pong (keepalives) and Text frames must be skipped, not turned into errors: a
        // single keepalive ping would otherwise look like a fatal error to the consuming loop
        // (e.g. the client's response demux) and kill an otherwise-healthy connection.
        // tungstenite already answers pings with pongs at the protocol layer, so ignoring them
        // here is safe. Mirrors the server channel's `recv`.
        loop {
            match self.inner.next().await {
                Some(Ok(tungstenite::Message::Binary(data))) => return Ok(Some(Message { data })),
                // A close frame is an orderly end-of-stream.
                Some(Ok(tungstenite::Message::Close(_close_frame))) => return Ok(None),
                // Control/text frames carry no protocol payload: skip and keep reading.
                Some(Ok(
                    tungstenite::Message::Ping(_)
                    | tungstenite::Message::Pong(_)
                    | tungstenite::Message::Text(_)
                    | tungstenite::Message::Frame(_),
                )) => continue,
                Some(Err(err)) => return Err(WsClientError::Tungstenite(err)),
                // The stream is exhausted: the peer went away without a close frame (server
                // shut the socket, network drop). A common disconnect path, not an error —
                // treat it as a clean end-of-stream rather than panicking with `todo!()`.
                None => return Ok(None),
            }
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
    /// The address couldn't be parsed (e.g. missing scheme, unsupported scheme).
    Address(String),
    Io(std::io::Error),
    Tungstenite(tungstenite::Error),
    UnknownMessage(String),
    Other(String),
}
impl CommunicationError for WsClientError {}

impl core::fmt::Display for WsClientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Address(msg) => write!(f, "invalid address: {msg}"),
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Tungstenite(err) => write!(f, "websocket error: {err}"),
            Self::UnknownMessage(msg) => write!(f, "unknown message: {msg}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for WsClientError {}

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
