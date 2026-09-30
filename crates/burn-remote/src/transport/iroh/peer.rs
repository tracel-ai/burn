use core::fmt;
use std::{net::SocketAddr, sync::Arc};

use iroh::{Endpoint, EndpointAddr, endpoint::BindError};
use tokio::sync::OnceCell;

use super::{node::RemoteNode, relays::IrohRelays};
use crate::{RemoteDevice, client::SessionOpenError, shared::SessionRefusal};

/// An Iroh compute server as a client dials it: its id and any addresses to try directly, the
/// relays it uses, and the credential its authorizer checks. Built with [`IrohPeerBuilder`].
///
/// A peer and its clones dial every device from one endpoint, so they share one connection to the
/// server. Unless the builder was given one, that endpoint is bound by the first
/// [`connect`](Self::connect) and runs on its runtime, which must outlive the peer.
#[derive(Clone, Debug)]
pub struct IrohPeer {
    addr: EndpointAddr,
    relays: IrohRelays,
    credential: Credential,
    node: Arc<OnceCell<RemoteNode>>,
}

impl IrohPeer {
    /// Device `device_index` of this server.
    ///
    /// # Errors
    ///
    /// See [`ConnectError`].
    pub async fn connect(&self, device_index: usize) -> Result<RemoteDevice, ConnectError> {
        // An application's endpoint may find the server through its own address lookup.
        let has_endpoint = self.node.initialized();
        if self.relays == IrohRelays::Disabled
            && self.addr.ip_addrs().next().is_none()
            && !has_endpoint
        {
            return Err(ConnectError::NoAddress);
        }
        let node = self
            .node
            .get_or_try_init(|| async {
                let endpoint = self.relays.endpoint_builder().bind().await?;
                Ok::<_, BindError>(RemoteNode::from_endpoint(endpoint))
            })
            .await
            .map_err(|source| ConnectError::Bind { source })?;
        let device = RemoteDevice::iroh_on_node(
            node.clone(),
            self.addr(),
            device_index,
            self.credential.0.clone(),
        );

        // The handshake blocks until the server answers, so it runs off the async workers.
        let connecting = device.clone();
        match tokio::task::spawn_blocking(move || connecting.try_connect()).await {
            Ok(Ok(())) => Ok(device),
            Ok(Err(err)) => Err(err.into()),
            Err(err) => match err.try_into_panic() {
                Ok(panic) => std::panic::resume_unwind(panic),
                Err(_) => Err(ConnectError::Interrupted),
            },
        }
    }

    fn addr(&self) -> EndpointAddr {
        match &self.relays {
            IrohRelays::Private { url } => self.addr.clone().with_relay_url(url.clone()),
            IrohRelays::Public | IrohRelays::Disabled => self.addr.clone(),
        }
    }
}

/// Builds an [`IrohPeer`]. Unless set otherwise, it is reached through n0's public relays with no
/// credential, from an endpoint Burn binds.
#[derive(Clone, Debug)]
pub struct IrohPeerBuilder {
    addr: EndpointAddr,
    relays: IrohRelays,
    credential: Credential,
    endpoint: Option<Endpoint>,
}

impl IrohPeerBuilder {
    /// The server at `server`: its id alone, or an [`EndpointAddr`] that also lists addresses.
    pub fn new(server: impl Into<EndpointAddr>) -> Self {
        Self {
            addr: server.into(),
            relays: IrohRelays::default(),
            credential: Credential::default(),
            endpoint: None,
        }
    }

    /// The relays the server uses.
    pub fn with_relays(mut self, relays: IrohRelays) -> Self {
        self.relays = relays;
        self
    }

    /// An address to try directly, required when relays are disabled. Give every address a host
    /// name resolves to: the server may listen on only one of IPv4 and IPv6.
    pub fn with_address(mut self, address: SocketAddr) -> Self {
        self.addr = self.addr.with_ip_addr(address);
        self
    }

    /// What the server's authorizer checks, such as the token of a server's `TokenAuthorizer`.
    pub fn with_credential(mut self, credential: impl Into<Vec<u8>>) -> Self {
        self.credential = Credential(credential.into());
        self
    }

    /// Dial from `endpoint`, shared with the application's other Iroh protocols, instead of
    /// binding one. Its own relay and segmentation offload settings then apply.
    pub fn with_endpoint(mut self, endpoint: Endpoint) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    /// Finish, ready to [`connect`](IrohPeer::connect).
    pub fn build(self) -> IrohPeer {
        IrohPeer {
            addr: self.addr,
            relays: self.relays,
            credential: self.credential,
            node: Arc::new(OnceCell::new_with(
                self.endpoint.map(RemoteNode::from_endpoint),
            )),
        }
    }
}

/// A credential that stays out of `Debug` output, since it is often a shared secret.
#[derive(Clone, Default)]
struct Credential(Vec<u8>);

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("..")
    }
}

/// Why [`IrohPeer::connect`] returned no device.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConnectError {
    /// Relays are disabled and no address was given, so an endpoint Burn binds cannot find the
    /// server.
    NoAddress,
    /// The local endpoint could not be bound.
    Bind {
        /// Iroh's reason.
        source: BindError,
    },
    /// The runtime shut down before the connection was attempted.
    Interrupted,
    /// No session could be opened with the server: it is not running, nothing answered at its
    /// addresses, or another endpoint answered there.
    Unreachable {
        /// What failed.
        reason: String,
    },
    /// The server's authorizer rejected the credential. Its reason is in the server's log.
    Unauthorized,
    /// The server does not host the device index asked for.
    NoSuchDevice {
        /// How many devices the server hosts.
        device_count: usize,
    },
    /// The server speaks another version of the Burn Remote protocol: build both with the same
    /// Burn release.
    IncompatibleProtocol,
    /// The server was reached, but the session handshake broke off or its reply made no sense. A
    /// server on an older Burn release refuses a session by closing it unanswered, which lands here.
    Handshake {
        /// What went wrong.
        reason: String,
    },
}

impl From<SessionOpenError> for ConnectError {
    fn from(err: SessionOpenError) -> Self {
        match err {
            SessionOpenError::Unreachable { reason } => Self::Unreachable { reason },
            SessionOpenError::Refused { refusal } => match refusal {
                SessionRefusal::Unauthorized => Self::Unauthorized,
                SessionRefusal::NoSuchDevice { device_count } => Self::NoSuchDevice {
                    device_count: device_count as usize,
                },
                SessionRefusal::IncompatibleProtocol => Self::IncompatibleProtocol,
            },
            SessionOpenError::Handshake { reason } => Self::Handshake { reason },
        }
    }
}

impl fmt::Display for ConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAddress => f.write_str("relays are disabled and no address was given"),
            Self::Bind { source } => write!(f, "cannot bind an Iroh endpoint: {source}"),
            Self::Interrupted => f.write_str("the runtime shut down before connecting"),
            Self::Unreachable { reason } => write!(f, "the server cannot be reached: {reason}"),
            Self::Unauthorized => f.write_str("the server's authorizer rejected the credential"),
            Self::NoSuchDevice { device_count } => {
                write!(f, "the server hosts only {device_count} device(s)")
            }
            Self::IncompatibleProtocol => {
                f.write_str("the server speaks another version of the Burn Remote protocol")
            }
            Self::Handshake { reason } => write!(f, "the session handshake failed: {reason}"),
        }
    }
}

impl std::error::Error for ConnectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind { source } => Some(source),
            Self::NoAddress
            | Self::Interrupted
            | Self::Unreachable { .. }
            | Self::Unauthorized
            | Self::NoSuchDevice { .. }
            | Self::IncompatibleProtocol
            | Self::Handshake { .. } => None,
        }
    }
}
