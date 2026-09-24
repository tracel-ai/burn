//! Process-level Iroh endpoint used by Burn Remote clients and compute nodes.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex, Weak},
};

#[cfg(feature = "server")]
use iroh::endpoint::VarInt;
use iroh::{
    Endpoint, EndpointAddr, EndpointId,
    endpoint::{Connection, RecvStream, SendStream},
};
use tokio::sync::OnceCell;

use crate::{PeerAddr, PeerId};

/// ALPN used by the version-one Burn Remote protocol.
pub const BURN_REMOTE_ALPN: &[u8] = b"burn/remote/1";

/// Identifies the purpose of a bidirectional stream inside a shared Iroh connection.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum StreamKind {
    Session,
    TensorTransfer,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StreamHeader {
    version: u16,
    kind: StreamKind,
}

const STREAM_VERSION: u16 = 1;
const MAX_FRAME_SIZE: usize = 1024 * 1024 * 1024;

/// The stream error a node answers with when it serves nothing: the opener's read fails at once.
#[cfg(feature = "server")]
const REFUSED: VarInt = VarInt::from_u32(1);

/// What a node answers the streams its peers open with, on every connection it holds.
#[cfg(feature = "server")]
pub(crate) trait Service: Send + Sync + 'static {
    fn serve(
        self: Arc<Self>,
        remote: EndpointId,
        kind: StreamKind,
        send: SendStream,
        recv: RecvStream,
    );
}

static NODES: LazyLock<Mutex<HashMap<EndpointId, Weak<RemoteNodeInner>>>> =
    LazyLock::new(Default::default);

struct RemoteNodeInner {
    endpoint: Endpoint,
    connections: tokio::sync::Mutex<HashMap<EndpointId, Arc<OnceCell<Connection>>>>,
    #[cfg(feature = "server")]
    service: Mutex<Option<Weak<dyn Service>>>,
}

/// Burn Remote on one Iroh endpoint: the devices a program drives from it and the server it hosts
/// on it.
///
/// A connection works both ways whichever side dialed it, so the node keeps one per peer and
/// answers what the peer opens on each with the service it hosts, or refuses when it hosts none.
/// Without the `server` feature a node hosts nothing and answers nothing.
#[derive(Clone)]
pub struct RemoteNode {
    inner: Arc<RemoteNodeInner>,
}

impl core::fmt::Debug for RemoteNode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RemoteNode")
            .field("endpoint_id", &self.id())
            .finish_non_exhaustive()
    }
}

impl RemoteNode {
    /// The node of `endpoint`, shared by every device and server in the process that uses an
    /// endpoint with its identity.
    ///
    /// On native client builds, the runtime that drives a device's session is captured when the
    /// device is created, not here, so create devices from the Tokio runtime that owns the
    /// endpoint.
    pub fn new(endpoint: &Endpoint) -> Self {
        let mut nodes = NODES.lock().unwrap();
        if let Some(inner) = nodes.get(&endpoint.id()).and_then(Weak::upgrade) {
            return Self { inner };
        }

        nodes.retain(|_, node| node.strong_count() > 0);
        let inner = Arc::new(RemoteNodeInner {
            endpoint: endpoint.clone(),
            connections: Default::default(),
            #[cfg(feature = "server")]
            service: Mutex::new(None),
        });
        nodes.insert(endpoint.id(), Arc::downgrade(&inner));
        Self { inner }
    }

    /// The cryptographic identity of this node.
    pub fn id(&self) -> EndpointId {
        self.inner.endpoint.id()
    }

    /// Access the underlying endpoint for relay, discovery, router, and observability setup.
    pub fn endpoint(&self) -> &Endpoint {
        &self.inner.endpoint
    }

    /// Answer the streams peers open on this node's connections with `service`, including the
    /// connections dialed before it. Held weakly, so the server it belongs to still drops.
    ///
    /// # Panics
    ///
    /// Panics when the node already hosts a service that is still alive: an endpoint serves one
    /// server.
    #[cfg(feature = "server")]
    pub(crate) fn host(&self, service: Weak<dyn Service>) {
        let mut hosted = self.inner.service.lock().unwrap();
        assert!(
            hosted
                .as_ref()
                .is_none_or(|current| current.strong_count() == 0),
            "an Iroh endpoint hosts one Burn Remote server, and this one already has one"
        );
        *hosted = Some(service);
    }

    /// Keep a connection a peer dialed for reuse, and answer what the peer opens on it until it
    /// closes.
    #[cfg(feature = "server")]
    pub(crate) async fn accept(&self, connection: Connection) -> Result<(), String> {
        self.remember(connection.clone()).await;
        answer(Arc::downgrade(&self.inner), connection).await
    }

    pub(crate) async fn open_stream(
        &self,
        peer: &PeerAddr,
        kind: StreamKind,
    ) -> Result<(SendStream, RecvStream), String> {
        // Only the Iroh variant remains when the websocket transport is compiled out.
        #[cfg_attr(
            not(feature = "websocket"),
            allow(clippy::infallible_destructuring_match)
        )]
        let peer = match peer {
            PeerAddr::Iroh(peer) => peer,
            #[cfg(feature = "websocket")]
            PeerAddr::WebSocket(_) => {
                return Err("Iroh node cannot open a stream to a non-Iroh peer".into());
            }
        };
        let connection = self.connection(peer.clone()).await?;
        let (mut send, recv) = connection
            .open_bi()
            .await
            .map_err(|err| format!("Failed to open Iroh stream to {}: {err}", peer.id))?;
        let header = rmp_serde::to_vec(&StreamHeader {
            version: STREAM_VERSION,
            kind,
        })
        .map_err(|err| format!("Failed to encode Iroh stream header: {err}"))?;
        send_frame(&mut send, &header).await?;
        Ok((send, recv))
    }

    async fn connection(&self, peer: EndpointAddr) -> Result<Connection, String> {
        loop {
            let cell = {
                let mut connections = self.inner.connections.lock().await;
                connections
                    .entry(peer.id)
                    .or_insert_with(|| Arc::new(OnceCell::new()))
                    .clone()
            };

            if let Some(connection) = cell.get()
                && connection.close_reason().is_some()
            {
                let mut connections = self.inner.connections.lock().await;
                if connections
                    .get(&peer.id)
                    .is_some_and(|current| Arc::ptr_eq(current, &cell))
                {
                    connections.remove(&peer.id);
                }
                continue;
            }

            let endpoint = self.inner.endpoint.clone();
            let peer_for_connect = peer.clone();
            #[cfg(feature = "server")]
            let node = Arc::downgrade(&self.inner);
            let connection = cell
                .get_or_try_init(|| async move {
                    let connection = endpoint
                        .connect(peer_for_connect.clone(), BURN_REMOTE_ALPN)
                        .await
                        .map_err(|err| {
                            format!(
                                "Failed to connect to Iroh peer {}: {err}",
                                peer_for_connect.id
                            )
                        })?;
                    // Runs once per connection, since the cell initializes once.
                    #[cfg(feature = "server")]
                    crate::server::spawn::spawn_detached(answer_dialed(node, connection.clone()));
                    Ok::<Connection, String>(connection)
                })
                .await?;
            return Ok(connection.clone());
        }
    }

    #[cfg(feature = "server")]
    async fn remember(&self, connection: Connection) {
        let remote = connection.remote_id();
        let cell = {
            let mut connections = self.inner.connections.lock().await;
            match connections.get(&remote) {
                Some(cell)
                    if cell
                        .get()
                        .is_some_and(|existing| existing.close_reason().is_none()) =>
                {
                    return;
                }
                _ => {
                    let cell = Arc::new(OnceCell::new());
                    connections.insert(remote, cell.clone());
                    cell
                }
            }
        };
        let _ = cell.set(connection);
    }
}

#[cfg(feature = "server")]
impl RemoteNodeInner {
    fn service(&self) -> Option<Arc<dyn Service>> {
        self.service
            .lock()
            .unwrap()
            .as_ref()
            .and_then(Weak::upgrade)
    }
}

/// Answer the streams opened on `connection` for as long as it and its node live.
#[cfg(feature = "server")]
async fn answer(node: Weak<RemoteNodeInner>, connection: Connection) -> Result<(), String> {
    let remote = connection.remote_id();
    while let Some((kind, send, recv)) = accept_stream(&connection).await? {
        let Some(node) = node.upgrade() else {
            refuse(remote, kind, send, recv);
            return Ok(());
        };
        match node.service() {
            Some(service) => service.serve(remote, kind, send, recv),
            None => refuse(remote, kind, send, recv),
        }
    }
    Ok(())
}

#[cfg(feature = "server")]
async fn answer_dialed(node: Weak<RemoteNodeInner>, connection: Connection) {
    let remote = connection.remote_id();
    if let Err(err) = answer(node, connection).await {
        log::warn!("Stream from {remote} on a connection we dialed failed: {err}");
    }
}

#[cfg(feature = "server")]
fn refuse(remote: EndpointId, kind: StreamKind, mut send: SendStream, mut recv: RecvStream) {
    log::warn!("Refused a {kind:?} stream from {remote}: this endpoint hosts no server");
    let _ = send.reset(REFUSED);
    let _ = recv.stop(REFUSED);
}

#[cfg(feature = "server")]
async fn accept_stream(
    connection: &Connection,
) -> Result<Option<(StreamKind, SendStream, RecvStream)>, String> {
    let (send, mut recv) = match connection.accept_bi().await {
        Ok(stream) => stream,
        Err(err) => {
            if connection.close_reason().is_some() {
                return Ok(None);
            }
            return Err(format!("Failed to accept Iroh stream: {err}"));
        }
    };
    let Some(frame) = recv_frame(&mut recv).await? else {
        return Ok(None);
    };
    let header: StreamHeader = rmp_serde::from_slice(&frame)
        .map_err(|err| format!("Invalid Iroh stream header: {err}"))?;
    if header.version != STREAM_VERSION {
        return Err(format!(
            "Unsupported Burn Remote stream version {} (expected {STREAM_VERSION})",
            header.version
        ));
    }
    Ok(Some((header.kind, send, recv)))
}

pub(crate) async fn send_frame(send: &mut SendStream, bytes: &[u8]) -> Result<(), String> {
    if bytes.len() > MAX_FRAME_SIZE {
        return Err(format!(
            "Burn Remote frame is too large: {} bytes (max {MAX_FRAME_SIZE})",
            bytes.len()
        ));
    }
    send.write_all(&(bytes.len() as u64).to_le_bytes())
        .await
        .map_err(|err| format!("Failed to write Iroh frame length: {err}"))?;
    send.write_all(bytes)
        .await
        .map_err(|err| format!("Failed to write Iroh frame: {err}"))?;
    Ok(())
}

pub(crate) async fn recv_frame(recv: &mut RecvStream) -> Result<Option<Vec<u8>>, String> {
    let Some(length) = recv_frame_length(recv).await? else {
        return Ok(None);
    };
    recv_frame_body(recv, length).await.map(Some)
}

/// The length prefix of the next frame, or `None` when the peer finished the stream.
pub(crate) async fn recv_frame_length(recv: &mut RecvStream) -> Result<Option<usize>, String> {
    let mut length = [0u8; 8];
    match recv.read_exact(&mut length).await {
        Ok(_) => {}
        Err(iroh::endpoint::ReadExactError::FinishedEarly(0)) => return Ok(None),
        Err(err) => return Err(format!("Failed to read Iroh frame length: {err}")),
    }
    let length = u64::from_le_bytes(length) as usize;
    if length > MAX_FRAME_SIZE {
        return Err(format!(
            "Peer sent an oversized Burn Remote frame: {length} bytes (max {MAX_FRAME_SIZE})"
        ));
    }
    Ok(Some(length))
}

/// The `length` bytes of a frame whose prefix [`recv_frame_length`] read.
pub(crate) async fn recv_frame_body(
    recv: &mut RecvStream,
    length: usize,
) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0; length];
    recv.read_exact(&mut bytes)
        .await
        .map_err(|err| format!("Failed to read Iroh frame: {err}"))?;
    Ok(bytes)
}

impl From<&RemoteNode> for PeerId {
    fn from(value: &RemoteNode) -> Self {
        PeerId::Iroh(value.id())
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use core::time::Duration;

    use iroh::{
        RelayMode,
        endpoint::presets,
        protocol::{AcceptError, ProtocolHandler, Router},
    };

    use super::*;

    #[derive(Debug, Clone)]
    struct Accepting(RemoteNode);

    impl ProtocolHandler for Accepting {
        async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
            self.0
                .accept(connection)
                .await
                .map_err(|err| AcceptError::from_err(std::io::Error::other(err)))
        }
    }

    async fn local_endpoint() -> Endpoint {
        Endpoint::builder(presets::Minimal)
            .relay_mode(RelayMode::Disabled)
            .clear_ip_transports()
            .bind_addr("127.0.0.1:0")
            .unwrap()
            .bind()
            .await
            .unwrap()
    }

    async fn refused(mut recv: RecvStream) -> bool {
        tokio::time::timeout(Duration::from_secs(10), recv_frame(&mut recv))
            .await
            .expect("a refusal, not silence")
            .is_err()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_node_hosting_nothing_refuses_streams_whoever_dialed() {
        let dialer = local_endpoint().await;
        let acceptor = local_endpoint().await;
        let acceptor_node = RemoteNode::new(&acceptor);
        let router = Router::builder(acceptor.clone())
            .accept(BURN_REMOTE_ALPN, Accepting(acceptor_node.clone()))
            .spawn();

        let dialer_node = RemoteNode::new(&dialer);
        let (_send, recv) = dialer_node
            .open_stream(&PeerAddr::Iroh(acceptor.addr()), StreamKind::Session)
            .await
            .unwrap();
        assert!(refused(recv).await);

        // The dialer's endpoint accepts no connections, so this stream can only ride the one it
        // dialed.
        let (_send, recv) = acceptor_node
            .open_stream(&PeerAddr::Iroh(dialer.addr()), StreamKind::TensorTransfer)
            .await
            .unwrap();
        assert!(refused(recv).await);

        router.shutdown().await.unwrap();
    }

    #[test]
    fn an_endpoint_has_one_node() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let endpoint = runtime.block_on(local_endpoint());

        let first = RemoteNode::new(&endpoint);
        let second = RemoteNode::new(&endpoint);

        assert!(Arc::ptr_eq(&first.inner, &second.inner));
    }
}
