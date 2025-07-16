use crate::{
    base::{Address, Protocol},
    websocket::{client::WsClient, server::WsServer},
};

#[derive(Clone)]
pub struct WsNetwork {}
impl Protocol for WsNetwork {
    type Client = WsClient;
    type Server = WsServer;
}

/// Parse an address, add the ws:// prefix if needed, and return an error if the address is invalid
pub fn parse_ws_address(mut address: Address) -> Result<Address, String> {
    let s = &address.inner;
    let parts = s.split("://").collect::<Vec<&str>>();
    let num_parts = parts.len();
    let url = if num_parts == 2 {
        if parts[0] == "ws" {
            s.to_owned()
        } else {
            return Err(format!("Invalid prefix: {}", parts[0]));
        }
    } else if num_parts == 1 {
        return Err(format!("ws://{s}"));
    } else {
        return Err(format!("Invalid url: {s}"));
    };

    address.inner = url;
    Ok(address)
}
