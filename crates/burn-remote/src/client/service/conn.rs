//! The client's transport connection layer.
//!
//! All `cfg(feature = ...)` transport selection on the client lives here: how to reach a peer
//! ([`RemoteEndpoint`]), the two halves of an opened session
//! ([`SubmitChannel`] / [`ResponseChannel`]), and the connect logic ([`RemoteEndpoint::open_channels`]). The
//! service and the registry are written against these and stay transport-agnostic.

use core::time::Duration;
use std::sync::Arc;

use crate::{
    PeerAddr, PeerId,
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

/// Everything needed to establish a session with a remote compute peer.
#[derive(Clone, Debug)]
pub(crate) enum RemoteEndpoint {
    #[cfg(feature = "iroh")]
    Iroh {
        node: RemoteNode,
        peer: iroh::EndpointAddr,
        authorization: Arc<[u8]>,
    },
    #[cfg(feature = "websocket")]
    WebSocket {
        address: Address,
        authorization: Arc<[u8]>,
    },
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

    pub(crate) fn authorization(&self) -> &[u8] {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh { authorization, .. } => authorization,
            #[cfg(feature = "websocket")]
            Self::WebSocket { authorization, .. } => authorization,
        }
    }

    /// The stable registry key for this endpoint (identity + authorization, no mutable dialing
    /// hints), so the same compute peer reuses one device id across reconnects.
    pub(crate) fn key(&self) -> EndpointKey {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh {
                node,
                peer,
                authorization,
                ..
            } => EndpointKey::Iroh {
                local: node.id(),
                remote: peer.id,
                authorization: authorization.clone(),
            },
            #[cfg(feature = "websocket")]
            Self::WebSocket {
                address,
                authorization,
                ..
            } => EndpointKey::WebSocket {
                address: address.clone(),
                authorization: authorization.clone(),
            },
        }
    }

    /// Open the session, returning its submit + response halves, and try again while the peer is not
    /// reachable yet.
    ///
    /// Done up front so a missing server surfaces here rather than on the first op, and the demux /
    /// writer tasks can be spawned on already-open streams.
    pub(crate) async fn open_channels(&self) -> Result<(SubmitChannel, ResponseChannel), String> {
        let peer = self.peer_id().to_short_string();
        let give_up = |err: OpenError| match err {
            OpenError::NotReachableYet(reason) => {
                let waited: Duration = OPEN_RETRY_DELAYS.iter().sum();
                format!(
                    "Cannot reach {peer} after trying for {waited:?} ({reason}). Is its server running?"
                )
            }
            OpenError::Failed(message) => {
                format!("Cannot open a remote session to {peer}: {message}")
            }
        };

        for delay in OPEN_RETRY_DELAYS {
            match self.open_channels_once().await {
                Err(OpenError::NotReachableYet(reason)) => {
                    log::info!("Cannot reach {peer} yet ({reason}), trying again in {delay:?}");
                    crate::time::sleep(delay).await;
                }
                opened => return opened.map_err(give_up),
            }
        }
        // The last attempt, with nothing left to wait for.
        self.open_channels_once().await.map_err(give_up)
    }

    async fn open_channels_once(&self) -> Result<(SubmitChannel, ResponseChannel), OpenError> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh { node, peer, .. } => {
                let (send, recv) = node
                    .open_stream(
                        &PeerAddr::Iroh(peer.clone()),
                        crate::transport::iroh::node::StreamKind::Session,
                    )
                    .await?;
                Ok((SubmitChannel::Iroh(send), ResponseChannel::Iroh(recv)))
            }
            #[cfg(feature = "websocket")]
            Self::WebSocket { address, .. } => {
                // One full-duplex socket per session, split into its submit (sink) and response
                // (source) halves, matching the Iroh single-stream model.
                let channel = WsClient::connect(address.clone(), "session").await?;
                let (sink, source) = channel.split();
                Ok((
                    SubmitChannel::WebSocket(Box::new(sink)),
                    ResponseChannel::WebSocket(Box::new(source)),
                ))
            }
        }
    }
}

/// Stable identity used to deduplicate endpoints in the device registry.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum EndpointKey {
    #[cfg(feature = "iroh")]
    Iroh {
        local: iroh::EndpointId,
        remote: iroh::EndpointId,
        authorization: Arc<[u8]>,
    },
    #[cfg(feature = "websocket")]
    WebSocket {
        address: Address,
        authorization: Arc<[u8]>,
    },
}

/// Outgoing half of an opened session (the submit stream).
pub(crate) enum SubmitChannel {
    #[cfg(feature = "iroh")]
    Iroh(iroh::endpoint::SendStream),
    #[cfg(feature = "websocket")]
    WebSocket(Box<burn_communication::websocket::WsClientSink>),
}

impl SubmitChannel {
    pub(crate) async fn send(&mut self, bytes: bytes::Bytes) -> Result<(), String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => FrameSink::send(stream, bytes).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(sink) => FrameSink::send(sink.as_mut(), bytes).await,
        }
    }

    pub(crate) async fn close(&mut self) -> Result<(), String> {
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

impl ResponseChannel {
    pub(crate) async fn recv(&mut self) -> Result<Option<bytes::Bytes>, String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => FrameSource::recv(stream).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(stream) => FrameSource::recv(stream.as_mut()).await,
        }
    }
}
