use std::time::Duration;

use socket2::{SockRef, TcpKeepalive};
use tokio::net::TcpStream;

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

/// Probe an idle connection after 10 s, then every 5 s, and drop it after 4 unanswered probes: a
/// vanished peer is noticed in 30 s, like iroh's idle timeout.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "windows",
))]
const KEEPALIVE: TcpKeepalive = TcpKeepalive::new()
    .with_time(Duration::from_secs(10))
    .with_interval(Duration::from_secs(5))
    .with_retries(4);

/// Elsewhere only the first probe's delay can be set; the interval and count are the system's.
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "windows",
)))]
const KEEPALIVE: TcpKeepalive = TcpKeepalive::new().with_time(Duration::from_secs(10));

/// Drops a connection whose peer vanished without closing it (a crash, a dropped network, a
/// sleeping laptop), which TCP alone never notices while the connection is idle.
pub(crate) trait DeadPeerTimeout {
    /// The connection works without it, so a failure is logged rather than returned.
    fn set_dead_peer_timeout(&self);
}

impl DeadPeerTimeout for TcpStream {
    fn set_dead_peer_timeout(&self) {
        if let Err(err) = SockRef::from(self).set_tcp_keepalive(&KEEPALIVE) {
            log::warn!("Cannot set TCP keep-alive, so a vanished peer will not be noticed: {err}");
        }
    }
}
