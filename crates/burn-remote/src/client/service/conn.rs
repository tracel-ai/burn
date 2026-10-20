//! The client's transport connection layer.
//!
//! All `cfg(feature = ...)` transport selection on the client lives here: how to reach a peer
//! ([`RemoteEndpoint`]), the two halves of an opened session
//! ([`SubmitChannel`] / [`ResponseChannel`]), and the connect logic ([`RemoteEndpoint::open_channels`]). The
//! service and the registry are written against these and stay transport-agnostic.

use core::time::Duration;

use super::OPEN_DEADLINE;
use crate::{
    ConnectError, Credential, PeerAddr, PeerId,
    transport::{
        OpenError,
        link::{FrameSink, FrameSource},
    },
};

#[cfg(feature = "iroh")]
use crate::transport::iroh::node::RemoteNode;
#[cfg(feature = "websocket")]
use burn_communication::{Address, ProtocolClient, websocket::WsClient};

/// How long to wait before each new attempt while the peer is not reachable yet. A server started
/// moments earlier has not opened its port, or published its address, which on n0's lookup takes
/// several seconds.
const OPEN_RETRY_DELAYS: [Duration; 6] = [
    Duration::from_millis(250),
    Duration::from_millis(500),
    Duration::from_secs(1),
    Duration::from_secs(2),
    Duration::from_secs(4),
    Duration::from_secs(8),
];

/// The two halves of an opened session.
pub(crate) struct SessionStreams {
    pub(crate) submit: SubmitChannel,
    pub(crate) response: ResponseChannel,
}

/// Everything needed to establish a session with a remote compute peer.
#[derive(Clone)]
pub(crate) enum RemoteEndpoint {
    #[cfg(feature = "iroh")]
    Iroh {
        node: RemoteNode,
        peer: iroh::EndpointAddr,
        credential: Credential,
        /// The application endpoint dialed from, or `None` for an endpoint Burn binds.
        app_endpoint: Option<iroh::EndpointId>,
    },
    #[cfg(feature = "websocket")]
    WebSocket {
        address: Address,
        credential: Credential,
    },
}

impl core::fmt::Debug for RemoteEndpoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never the credential, which is often a shared secret.
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh { node, peer, .. } => f
                .debug_struct("Iroh")
                .field("node", node)
                .field("peer", peer)
                .finish_non_exhaustive(),
            #[cfg(feature = "websocket")]
            Self::WebSocket { address, .. } => f
                .debug_struct("WebSocket")
                .field("address", address)
                .finish_non_exhaustive(),
        }
    }
}

impl RemoteEndpoint {
    pub(crate) fn peer_addr(&self) -> PeerAddr {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh { peer, .. } => PeerAddr::Iroh(peer.clone()),
            #[cfg(feature = "websocket")]
            Self::WebSocket { address, .. } => PeerAddr::WebSocket(address.clone()),
        }
    }

    pub(crate) fn peer_id(&self) -> PeerId {
        self.peer_addr().id()
    }

    pub(crate) fn credential(&self) -> &Credential {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh { credential, .. } => credential,
            #[cfg(feature = "websocket")]
            Self::WebSocket { credential, .. } => credential,
        }
    }

    /// The stable registry key for this endpoint, without dialing hints. An endpoint Burn binds
    /// is left out, so one server reached under two relay settings is one device; an
    /// application endpoint stays in, so a second identity never inherits the first one's session.
    pub(crate) fn key(&self) -> EndpointKey {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh {
                peer,
                credential,
                app_endpoint,
                ..
            } => EndpointKey::Iroh {
                app_endpoint: *app_endpoint,
                remote: peer.id,
                credential: credential.clone(),
            },
            #[cfg(feature = "websocket")]
            Self::WebSocket {
                address,
                credential,
            } => EndpointKey::WebSocket {
                address: address.clone(),
                credential: credential.clone(),
            },
        }
    }

    /// Open the session, returning its submit + response halves, and try again while the peer is not
    /// reachable yet.
    ///
    /// Done up front so a missing server surfaces here rather than on the first op, and the demux /
    /// writer tasks can be spawned on already-open streams.
    pub(crate) async fn open_channels(&self) -> Result<SessionStreams, ConnectError> {
        let peer = self.peer_id().to_short_string();
        let give_up = |err: OpenError| match err {
            OpenError::NotReachableYet(reason) => {
                let waited: Duration = OPEN_RETRY_DELAYS.iter().sum();
                ConnectError::Unreachable {
                    reason: format!(
                        "nothing answered after trying for {waited:?} ({reason}); is the server running?"
                    ),
                }
            }
            #[cfg(feature = "iroh")]
            OpenError::NoAddress => ConnectError::NoAddress,
            OpenError::Failed(reason) => ConnectError::Unreachable { reason },
        };

        for delay in OPEN_RETRY_DELAYS {
            match self.open_channels_within_deadline().await {
                Err(OpenError::NotReachableYet(reason)) => {
                    log::info!("Cannot reach {peer} yet ({reason}), trying again in {delay:?}");
                    crate::time::sleep(delay).await;
                }
                opened => return opened.map_err(give_up),
            }
        }
        // The last attempt, with nothing left to wait for.
        self.open_channels_within_deadline().await.map_err(give_up)
    }

    /// One attempt, given up at the deadline on a server that accepts and never serves it.
    async fn open_channels_within_deadline(&self) -> Result<SessionStreams, OpenError> {
        crate::time::timeout(OPEN_DEADLINE, self.open_channels_once())
            .await
            .unwrap_or_else(|()| {
                Err(OpenError::Failed(format!(
                    "no connection opened within {OPEN_DEADLINE:?}"
                )))
            })
    }

    async fn open_channels_once(&self) -> Result<SessionStreams, OpenError> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh { node, peer, .. } => {
                let (send, recv) = node
                    .open_stream(
                        &PeerAddr::Iroh(peer.clone()),
                        crate::transport::iroh::node::StreamKind::Session,
                    )
                    .await?;
                Ok(SessionStreams {
                    submit: SubmitChannel::Iroh(send),
                    response: ResponseChannel::Iroh(recv),
                })
            }
            #[cfg(feature = "websocket")]
            Self::WebSocket { address, .. } => {
                // One full-duplex socket per session, split into its submit (sink) and response
                // (source) halves, matching the Iroh single-stream model.
                let channel = WsClient::connect(address.clone(), "session").await?;
                let (sink, source) = channel.split();
                Ok(SessionStreams {
                    submit: SubmitChannel::WebSocket(Box::new(sink)),
                    response: ResponseChannel::WebSocket(Box::new(source)),
                })
            }
        }
    }
}

/// Stable identity used to deduplicate endpoints in the device registry.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum EndpointKey {
    #[cfg(feature = "iroh")]
    Iroh {
        app_endpoint: Option<iroh::EndpointId>,
        remote: iroh::EndpointId,
        credential: Credential,
    },
    #[cfg(feature = "websocket")]
    WebSocket {
        address: Address,
        credential: Credential,
    },
}

/// Outgoing half of an opened session (the submit stream).
pub(crate) enum SubmitChannel {
    #[cfg(feature = "iroh")]
    Iroh(iroh::endpoint::SendStream),
    #[cfg(feature = "websocket")]
    WebSocket(Box<burn_communication::websocket::WsClientSink>),
}

impl FrameSink for SubmitChannel {
    async fn send(&mut self, bytes: bytes::Bytes) -> Result<(), String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => FrameSink::send(stream, bytes).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(sink) => FrameSink::send(sink.as_mut(), bytes).await,
        }
    }

    async fn close(&mut self) -> Result<(), String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => FrameSink::close(stream).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(sink) => FrameSink::close(sink.as_mut()).await,
        }
    }
}

/// Incoming half of an opened session (the response stream).
pub(crate) enum ResponseChannel {
    #[cfg(feature = "iroh")]
    Iroh(iroh::endpoint::RecvStream),
    #[cfg(feature = "websocket")]
    WebSocket(Box<burn_communication::websocket::WsClientStream>),
}

impl FrameSource for ResponseChannel {
    async fn recv(&mut self, max_len: usize) -> Result<Option<bytes::Bytes>, String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => FrameSource::recv(stream, max_len).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(stream) => FrameSource::recv(stream.as_mut(), max_len).await,
        }
    }

    async fn recv_into(&mut self, buf: &mut [u8]) -> Result<usize, String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => FrameSource::recv_into(stream, buf).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(stream) => FrameSource::recv_into(stream.as_mut(), buf).await,
        }
    }
}
