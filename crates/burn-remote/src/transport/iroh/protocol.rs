//! Iroh protocol handler for Burn Remote compute and tensor-transfer streams.

use std::{
    fmt,
    sync::{Arc, Weak},
};

use burn_backend::tensor::Device;
use burn_ir::BackendIr;
use burn_router::CustomOpRegistry;
use iroh::{
    Endpoint, EndpointId,
    endpoint::{Connection, RecvStream, SendStream},
    protocol::{AcceptError, DynProtocolHandler, ProtocolHandler},
};

use crate::{
    PeerId,
    server::{pump::drive_session, session::SessionManager, spawn::spawn_detached},
    telemetry::TelemetryProbe,
};

use super::{
    IrohTransfer,
    node::{RemoteNode, Service, StreamKind},
};

/// Information presented to a compute node before a remote session is accepted.
pub struct AuthorizationRequest<'a> {
    /// Authenticated Iroh identity of the connecting peer.
    pub peer: EndpointId,
    /// Compute-device index requested by the peer.
    pub device_index: u32,
    /// Opaque credential supplied by the application when creating the remote device.
    pub credential: &'a [u8],
}

/// Application authorization policy for incoming compute sessions.
pub trait PeerAuthorizer: Send + Sync + 'static {
    /// Return `Ok(())` to allow the session, or a user-facing rejection reason.
    fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<(), String>;
}

impl<F> PeerAuthorizer for F
where
    F: Fn(AuthorizationRequest<'_>) -> Result<(), String> + Send + Sync + 'static,
{
    fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<(), String> {
        self(request)
    }
}

#[derive(Debug, Default)]
pub struct AllowAll;

impl PeerAuthorizer for AllowAll {
    fn authorize(&self, _request: AuthorizationRequest<'_>) -> Result<(), String> {
        Ok(())
    }
}

/// Iroh protocol handler for Burn Remote compute and tensor-transfer streams.
///
/// Register this handler in an existing Iroh `Router` to compose Burn with other application
/// protocols on the same endpoint.
pub struct IrohRemoteProtocol<B: BackendIr> {
    service: Arc<ComputeService<B>>,
}

/// The sessions and tensor transfers a compute node serves, on every connection its node holds.
struct ComputeService<B: BackendIr> {
    node: RemoteNode,
    sessions: Arc<SessionManager<B, IrohTransfer<B>>>,
    transfer: Arc<IrohTransfer<B>>,
    authorizer: Arc<dyn PeerAuthorizer>,
}

impl<B: BackendIr> fmt::Debug for IrohRemoteProtocol<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IrohRemoteProtocol")
            .field("endpoint_id", &self.service.node.id())
            .finish_non_exhaustive()
    }
}

impl<B: BackendIr> IrohRemoteProtocol<B> {
    /// Create a handler hosting `devices` on `endpoint`.
    ///
    /// Anything hosting this runs on an async runtime, so it says so: a session's tensor read then
    /// materializes eagerly instead of parking a blocking device to host copy on an executor worker.
    /// Logging stays the application's, see [`ServerLogging`](crate::server::ServerLogging).
    ///
    /// # Panics
    ///
    /// Panics when another handler created on `endpoint` is still alive: an endpoint hosts one
    /// server.
    pub fn new(
        endpoint: Endpoint,
        devices: Vec<Device<B>>,
        authorizer: Arc<dyn PeerAuthorizer>,
        probe: TelemetryProbe,
        custom_ops: CustomOpRegistry<B>,
    ) -> Self {
        burn_std::set_runtime_kind(burn_std::RuntimeKind::Async);
        let node = RemoteNode::new(&endpoint);
        let transfer = Arc::new(IrohTransfer::new(node.clone()));
        let sessions = Arc::new(
            SessionManager::new(devices, transfer.clone())
                .with_telemetry(probe)
                .with_custom_ops(custom_ops),
        );
        let service = Arc::new(ComputeService {
            node,
            sessions,
            transfer,
            authorizer,
        });
        service
            .node
            .host(Arc::downgrade(&service) as Weak<dyn Service>);
        Self { service }
    }
}

impl<B: BackendIr> ComputeService<B> {
    /// Drive a session stream through the shared [`drive_session`] pump.
    ///
    /// The Iroh-specific parts are just the authenticated peer identity (`remote`, checked by the
    /// application's [`PeerAuthorizer`]) and this server's own id, echoed to the client.
    async fn session(
        &self,
        remote: EndpointId,
        send: SendStream,
        recv: RecvStream,
    ) -> Result<(), String> {
        drive_session(
            recv,
            send,
            self.sessions.clone(),
            Some(PeerId::Iroh(self.node.id())),
            |init| {
                self.authorizer.authorize(AuthorizationRequest {
                    peer: remote,
                    device_index: init.device_index,
                    credential: &init.authorization,
                })
            },
        )
        .await
    }
}

impl<B: BackendIr> Service for ComputeService<B> {
    fn serve(
        self: Arc<Self>,
        remote: EndpointId,
        kind: StreamKind,
        send: SendStream,
        recv: RecvStream,
    ) {
        match kind {
            StreamKind::Session => spawn_detached(async move {
                if let Err(err) = self.session(remote, send, recv).await {
                    log::warn!("Rejected or failed Iroh remote session: {err}");
                }
            }),
            StreamKind::TensorTransfer => spawn_detached(async move {
                if let Err(err) = self.transfer.handle_stream(remote, send, recv).await {
                    log::warn!("Iroh tensor-transfer stream failed: {err}");
                }
            }),
        }
    }
}

/// A backend-erased Burn Remote protocol handler.
///
/// The dispatch layer resolves a `Device` to a concrete backend and builds an
/// [`IrohRemoteProtocol`]; this wraps it as a single non-generic type, so an application can
/// register Burn on its own Iroh `Router` without naming a backend. Hand it directly to
/// `RouterBuilder::accept` under [`BURN_REMOTE_ALPN`](super::node::BURN_REMOTE_ALPN).
pub struct RemoteProtocol(Box<dyn DynProtocolHandler>);

impl RemoteProtocol {
    /// Erase a concrete protocol handler behind this non-generic type.
    pub fn new(handler: impl ProtocolHandler) -> Self {
        Self(handler.into())
    }
}

impl fmt::Debug for RemoteProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteProtocol").finish_non_exhaustive()
    }
}

impl From<RemoteProtocol> for Box<dyn DynProtocolHandler> {
    fn from(protocol: RemoteProtocol) -> Self {
        protocol.0
    }
}

impl<B: BackendIr> ProtocolHandler for IrohRemoteProtocol<B> {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        self.service
            .node
            .accept(connection)
            .await
            .map_err(user_error)
    }
}

fn user_error(reason: String) -> AcceptError {
    AcceptError::from_err(std::io::Error::other(reason))
}
