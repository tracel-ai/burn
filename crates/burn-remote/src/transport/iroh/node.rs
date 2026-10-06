//! Process-level Iroh endpoint used by Burn Remote clients and compute nodes.

use std::{collections::HashMap, sync::Arc};

#[cfg(feature = "client")]
use iroh::endpoint::BindError;
use iroh::{
    Endpoint, EndpointAddr, EndpointId,
    endpoint::{Connection, RecvStream, SendStream},
};
#[cfg(feature = "client")]
use std::sync::{LazyLock, Weak};
use tokio::sync::Mutex;
use tokio::sync::OnceCell;

#[cfg(feature = "client")]
use super::relays::IrohRelays;
use crate::{PeerAddr, PeerId, transport::OpenError};

/// The node the devices dialed from each application endpoint share, by its id. Weak, because Iroh
/// keeps an endpoint's sockets bound until its last clone drops.
#[cfg(feature = "client")]
static APP_NODES: LazyLock<std::sync::Mutex<HashMap<EndpointId, Weak<RemoteNodeInner>>>> =
    LazyLock::new(Default::default);

/// The node Burn binds for each relay setting, shared by every host that dials with it.
#[cfg(feature = "client")]
static OWNED_NODES: LazyLock<std::sync::Mutex<HashMap<IrohRelays, Arc<OnceCell<RemoteNode>>>>> =
    LazyLock::new(Default::default);

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

/// Changing it, or [`BURN_REMOTE_ALPN`], drops an older client before it can be told that its
/// protocol version differs.
const STREAM_VERSION: u16 = 1;
const MAX_FRAME_SIZE: usize = 1024 * 1024 * 1024;

struct RemoteNodeInner {
    endpoint: Endpoint,
    /// Only connections this node dialed: a peer answers no streams on a connection it dialed.
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
    /// A node of its own on `endpoint`, shared with nothing else in the process.
    pub fn from_endpoint(endpoint: Endpoint) -> Self {
        Self {
            inner: Arc::new(RemoteNodeInner {
                endpoint,
                connections: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// The node shared by every device dialed from `endpoint`.
    ///
    /// Iroh lets two live endpoints share one secret key, and a node keyed by that id would hand
    /// the second the first one's connections, so a second live endpoint is refused. A browser
    /// endpoint has no bound sockets to tell the two apart by, so there the second shares the
    /// first one's node. A closed endpoint's node is replaced.
    #[cfg(feature = "client")]
    pub(crate) fn for_endpoint(endpoint: &Endpoint) -> Result<Self, String> {
        let mut nodes = APP_NODES.lock().unwrap();
        if let Some(inner) = nodes.get(&endpoint.id()).and_then(Weak::upgrade)
            && !inner.endpoint.is_closed()
        {
            let node = Self { inner };
            #[cfg(not(target_family = "wasm"))]
            if node.endpoint().bound_sockets() != endpoint.bound_sockets() {
                return Err(format!(
                    "another open Iroh endpoint has the id {}; bind one endpoint per secret key",
                    endpoint.id().fmt_short()
                ));
            }
            return Ok(node);
        }
        nodes.retain(|_, inner| inner.strong_count() > 0);
        let node = Self::from_endpoint(endpoint.clone());
        nodes.insert(endpoint.id(), Arc::downgrade(&node.inner));
        Ok(node)
    }

    /// The node Burn binds for `relays`, bound the first time any host needs it.
    #[cfg(feature = "client")]
    pub(crate) async fn for_relays(relays: &IrohRelays) -> Result<Self, BindError> {
        let cell = OWNED_NODES
            .lock()
            .unwrap()
            .entry(relays.clone())
            .or_default()
            .clone();
        cell.get_or_try_init(|| async {
            let endpoint = relays.endpoint_builder().bind().await?;
            Ok(Self::from_endpoint(endpoint))
        })
        .await
        .cloned()
    }

    /// The cryptographic identity of this node.
    pub fn id(&self) -> EndpointId {
        self.inner.endpoint.id()
    }

    /// Access the underlying endpoint for relay, discovery, router, and observability setup.
    #[cfg(not(target_family = "wasm"))]
    pub fn endpoint(&self) -> &Endpoint {
        &self.inner.endpoint
    }

    pub(crate) async fn open_stream(
        &self,
        peer: &PeerAddr,
        kind: StreamKind,
    ) -> Result<(SendStream, RecvStream), OpenError> {
        // Only the Iroh variant remains when the websocket transport is compiled out.
        #[cfg_attr(
            not(feature = "websocket"),
            allow(clippy::infallible_destructuring_match)
        )]
        let peer = match peer {
            PeerAddr::Iroh(peer) => peer,
            #[cfg(feature = "websocket")]
            PeerAddr::WebSocket(_) => {
                return Err(OpenError::Failed(
                    "an Iroh node cannot open a stream to a non-Iroh peer".into(),
                ));
            }
        };
        let connection = self.connection(peer.clone()).await?;
        let (mut send, recv) = connection
            .open_bi()
            .await
            .map_err(|err| OpenError::Failed(format!("cannot open an Iroh stream: {err}")))?;
        let header = rmp_serde::to_vec(&StreamHeader {
            version: STREAM_VERSION,
            kind,
        })
        .map_err(|err| OpenError::Failed(format!("cannot encode the Iroh stream header: {err}")))?;
        send_frame(&mut send, &header)
            .await
            .map_err(OpenError::Failed)?;
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

    async fn connection(&self, peer: EndpointAddr) -> Result<Connection, OpenError> {
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
                        .map_err(OpenError::from)
                })
                .await?;
            return Ok(connection.clone());
        }
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
