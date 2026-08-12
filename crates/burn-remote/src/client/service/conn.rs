//! The client's transport connection layer.
//!
//! All `cfg(feature = ...)` transport selection on the client lives here: how to reach a peer
//! ([`RemoteEndpoint`]), the two halves of an opened session
//! ([`SubmitChannel`] / [`ResponseChannel`]), and the connect logic ([`open_channels`]). The
//! service and the registry are written against these and stay transport-agnostic.

use std::sync::Arc;

use crate::transport::link::{FrameSink, FrameSource};
use crate::{PeerAddr, PeerId};

#[cfg(feature = "iroh")]
use crate::transport::iroh::node::RemoteNode;
#[cfg(feature = "websocket")]
use burn_communication::{Address, ProtocolClient};

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

/// Open the session for `endpoint`, returning its submit + response halves.
///
/// Done up front so a missing server surfaces here rather than on the first op, and the demux /
/// writer tasks can be spawned on already-open streams.
pub(crate) async fn open_channels(
    endpoint: &RemoteEndpoint,
) -> Result<(SubmitChannel, ResponseChannel), String> {
    match endpoint {
        #[cfg(feature = "iroh")]
        RemoteEndpoint::Iroh { node, peer, .. } => {
            let (send, recv) = node
                .open_stream(
                    &PeerAddr::Iroh(peer.clone()),
                    crate::transport::iroh::node::StreamKind::Session,
                )
                .await?;
            Ok((SubmitChannel::Iroh(send), ResponseChannel::Iroh(recv)))
        }
        #[cfg(feature = "websocket")]
        RemoteEndpoint::WebSocket { address, .. } => {
            use burn_communication::websocket::WsClient;
            // One full-duplex socket per session, split into the submit (sink) + response (source)
            // halves — matching the Iroh single-stream model.
            let channel = WsClient::connect(address.clone(), "session")
                .await
                .map_err(|err| connect_error("session", &endpoint.peer_addr(), &err))?;
            let (sink, source) = channel.split();
            Ok((
                SubmitChannel::WebSocket(Box::new(sink)),
                ResponseChannel::WebSocket(Box::new(source)),
            ))
        }
    }
}

/// Actionable panic message for a failed channel connect.
#[cfg(feature = "websocket")]
fn connect_error<E: std::fmt::Debug>(route: &str, peer: &PeerAddr, err: &E) -> String {
    format!(
        "Failed to open remote '{route}' channel to {peer}: {err:?}. \
         Is a `burn-remote` compute node running at that peer?"
    )
}
