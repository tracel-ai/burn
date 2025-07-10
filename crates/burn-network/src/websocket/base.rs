use std::str::FromStr;

use crate::{
    network::Network,
    websocket::{client::WsClient, server::WsServer},
};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct WsNetwork {}
impl Network for WsNetwork {
    type Client = WsClient;
    type Server = WsServer;
}

/// Allows nodes to find each other
/// TODO url validation and shouldn't be ws spesific
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WsAddress {
    /// A url that includes the port and the ws:// prefix.
    inner: String,
}

impl FromStr for WsAddress {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
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

        Ok(Self { inner: url })
    }
}
