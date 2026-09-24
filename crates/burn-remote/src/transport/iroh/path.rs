//! Which network path an Iroh connection sends on: direct, or through a relay.

use core::{fmt, time::Duration};

use futures_util::StreamExt;
use iroh::{
    TransportAddr,
    endpoint::{Connection, PathList},
};

use crate::spawn::spawn_detached;

/// The path iroh selected to send a connection's data on.
#[derive(Debug)]
pub(crate) struct SelectedPath {
    remote: TransportAddr,
    rtt: Duration,
}

impl SelectedPath {
    /// The selected path among `paths`, `None` until iroh has selected one.
    pub(crate) fn of(paths: &PathList<'_>) -> Option<Self> {
        paths
            .iter()
            .find(|path| path.is_selected())
            .map(|path| Self {
                remote: path.remote_addr().clone(),
                rtt: path.rtt(),
            })
    }

    /// Log the path `connection` sends on now and whenever iroh switches it, until it closes. A
    /// relayed connection is slower than a direct one, and nothing else says which one a run got.
    pub(crate) fn log_changes(connection: &Connection) {
        let peer = connection.remote_id().fmt_short();
        // Subscribed before the first read, so a switch between the two is not missed.
        let mut events = connection.path_events();
        // Weak, so logging never keeps a connection open.
        let connection = connection.weak_handle();

        spawn_detached(async move {
            let mut logged = None;
            loop {
                let Some(current) = connection.upgrade() else {
                    return;
                };
                let selected = Self::of(&current.paths());
                drop(current);

                if let Some(path) = selected
                    && logged.as_ref() != Some(&path.remote)
                {
                    log::info!("Iroh connection to {peer} sends {path}");
                    logged = Some(path.remote);
                }
                if events.next().await.is_none() {
                    return;
                }
            }
        });
    }
}

impl fmt::Display for SelectedPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rtt = self.rtt.as_millis();
        match &self.remote {
            TransportAddr::Ip(addr) => write!(f, "direct to {addr}, RTT {rtt} ms"),
            TransportAddr::Relay(url) => write!(f, "through relay {url}, RTT {rtt} ms"),
            other => write!(f, "over {other}, RTT {rtt} ms"),
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use iroh::{Endpoint, RelayMode, endpoint::presets};

    use super::*;

    const ALPN: &[u8] = b"burn/remote/path-test";

    #[tokio::test(flavor = "multi_thread")]
    async fn a_loopback_connection_without_relays_sends_direct() {
        let server = local_endpoint().await;
        let client = local_endpoint().await;
        let address = server.addr();
        let accepted = tokio::spawn(async move {
            let connection = server.accept().await.unwrap().await.unwrap();
            connection.closed().await;
        });

        let connection = client.connect(address, ALPN).await.unwrap();
        let path = SelectedPath::of(&connection.paths()).expect("a connected path is selected");

        assert!(path.remote.is_ip(), "expected a direct path, got {path}");
        connection.close(0u32.into(), b"done");
        accepted.await.unwrap();
    }

    async fn local_endpoint() -> Endpoint {
        Endpoint::builder(presets::Minimal)
            .relay_mode(RelayMode::Disabled)
            .clear_ip_transports()
            .bind_addr("127.0.0.1:0")
            .unwrap()
            .alpns(vec![ALPN.to_vec()])
            .bind()
            .await
            .unwrap()
    }
}
