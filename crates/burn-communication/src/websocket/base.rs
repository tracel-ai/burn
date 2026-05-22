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

/// Parse an address, add the `ws://` prefix if missing, and return an error if the address
/// is invalid.
pub(crate) fn parse_ws_address(mut address: Address) -> Result<Address, String> {
    let s = &address.inner;
    let parts = s.split("://").collect::<Vec<&str>>();
    let url = match parts.as_slice() {
        [host] => format!("ws://{host}"),
        ["ws", _] => s.to_owned(),
        [prefix, _] => return Err(format!("Invalid prefix: {prefix}")),
        _ => return Err(format!("Invalid url: {s}")),
    };

    address.inner = url;
    Ok(address)
}
