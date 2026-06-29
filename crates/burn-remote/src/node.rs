//! Process-level Iroh endpoint used by Burn Remote clients and compute nodes.

use std::{collections::HashMap, sync::Arc};

use iroh::{
    Endpoint, EndpointAddr, EndpointId,
    endpoint::{Connection, RecvStream, SendStream},
};
use tokio::sync::Mutex;
use tokio::sync::OnceCell;

use crate::peer::{PeerAddr, PeerId};

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

struct RemoteNodeInner {
    endpoint: Endpoint,
    connections: Mutex<HashMap<EndpointId, Arc<OnceCell<Connection>>>>,
}

/// A process-level Burn Remote networking node.
///
/// Clone this handle freely. Every clone shares one Iroh [`Endpoint`] and one connection pool,
/// so all remote devices in the process multiplex their sessions over the same peer connection.
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
    /// Use an application-configured Iroh endpoint.
    ///
    /// Applications serving Burn Remote must include [`BURN_REMOTE_ALPN`] in the endpoint's
    /// accepted ALPN list, or route that ALPN to Burn's protocol handler.
    ///
    /// On native client builds, the runtime that drives a device's session is captured when the
    /// device is created (see [`RemoteNode::device`]), not here — so create devices from the
    /// Tokio runtime that owns this endpoint.
    pub fn from_endpoint(endpoint: Endpoint) -> Self {
        Self {
            inner: Arc::new(RemoteNodeInner {
                endpoint,
                connections: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// The cryptographic identity of this node.
    pub fn id(&self) -> EndpointId {
        self.inner.endpoint.id()
    }

    /// Access the underlying endpoint for relay, discovery, router, and observability setup.
    pub fn endpoint(&self) -> &Endpoint {
        &self.inner.endpoint
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

    #[cfg(feature = "server")]
    pub(crate) async fn accept_stream(
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
            let connection = cell
                .get_or_try_init(|| async move {
                    endpoint
                        .connect(peer_for_connect.clone(), BURN_REMOTE_ALPN)
                        .await
                        .map_err(|err| {
                            format!(
                                "Failed to connect to Iroh peer {}: {err}",
                                peer_for_connect.id
                            )
                        })
                })
                .await?;
            return Ok(connection.clone());
        }
    }

    #[cfg(feature = "server")]
    pub(crate) async fn remember_connection(&self, connection: Connection) {
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
    let mut bytes = vec![0; length];
    recv.read_exact(&mut bytes)
        .await
        .map_err(|err| format!("Failed to read Iroh frame: {err}"))?;
    Ok(Some(bytes))
}

impl From<&RemoteNode> for PeerId {
    fn from(value: &RemoteNode) -> Self {
        PeerId::Iroh(value.id())
    }
}
