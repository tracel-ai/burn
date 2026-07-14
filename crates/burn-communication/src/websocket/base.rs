use crate::{
    base::{Address, Protocol},
    websocket::{client::WsClient, server::WsServer},
};

#[derive(Clone)]
/// A websocket implements a [communication protocol](Protocol) that can be used to communicate
/// over the internet.
pub struct WebSocket {}

impl Protocol for WebSocket {
    type Client = WsClient;
    type Server = WsServer;
}

/// Validate that an [`Address`] uses the websocket scheme.
///
/// The [`Address`] is already canonicalized at construction (scheme defaults to `ws`, path
/// stripped), so this only has to reject a non-`ws` scheme. The address is returned
/// unchanged on success and its [`Display`](std::fmt::Display) form is the connection url.
pub(crate) fn parse_ws_address(address: Address) -> Result<Address, String> {
    match address.scheme() {
        "ws" | "wss" => Ok(address),
        other => Err(format!("Invalid scheme: {other}")),
    }
}
