use std::str::FromStr;

use crate::{
    network::{Network, NetworkAddress},
    websocket::{client::WsClient, server::WsServer},
};

#[derive(Clone)]
pub struct WsNetwork {}
impl Network for WsNetwork {
    type Client = WsClient;
    type Server = WsServer;
}

pub struct WsAddress(NetworkAddress);

impl FromStr for WsAddress {
    type Err = String;

    fn from_str(s: &str) -> Result<WsAddress, String> {
        let parts = s.split("://").collect::<Vec<&str>>();
        let num_parts = parts.len();
        let url = if num_parts == 2 {
            if parts[0] == "ws" {
                s.to_owned()
            } else {
                panic!("Invalid prefix: {}", parts[0]);
            }
        } else if num_parts == 1 {
            format!("ws://{s}")
        } else {
            panic!("Invalid url: {s}");
        };

        NetworkAddress::from_str(s).map(Self)
    }
}
