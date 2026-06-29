use std::{fmt, sync::Arc};

use burn_backend::tensor::Device;
use burn_ir::BackendIr;
use burn_router::CustomOpRegistry;
use iroh::{
    Endpoint, EndpointId,
    endpoint::{Connection, RecvStream, SendStream},
    protocol::{AcceptError, DynProtocolHandler, ProtocolHandler, Router},
};

use crate::{
    BURN_REMOTE_ALPN, PeerId,
    server::{
        pump::drive_session,
        session::SessionManager,
        spawn::{os_shutdown_signal, spawn_detached},
    },
    telemetry::TelemetryProbe,
};

use super::{
    IrohTransfer,
    node::{RemoteNode, StreamKind},
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
/// Register this handler in an existing Iroh [`Router`] to compose Burn with other application
/// protocols on the same endpoint.
pub struct IrohRemoteProtocol<B: BackendIr> {
    node: RemoteNode,
    sessions: Arc<SessionManager<B, IrohTransfer<B>>>,
    transfer: Arc<IrohTransfer<B>>,
    authorizer: Arc<dyn PeerAuthorizer>,
}

impl<B: BackendIr> fmt::Debug for IrohRemoteProtocol<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IrohRemoteProtocol")
            .field("endpoint_id", &self.node.id())
            .finish_non_exhaustive()
    }
}

impl<B: BackendIr> IrohRemoteProtocol<B> {
    /// Create a handler hosting `devices` on `node`.
    pub fn new(
        endpoint: Endpoint,
        devices: Vec<Device<B>>,
        authorizer: Arc<dyn PeerAuthorizer>,
        probe: TelemetryProbe,
        custom_ops: CustomOpRegistry<B>,
    ) -> Self {
        let node = RemoteNode::from_endpoint(endpoint);
        let transfer = Arc::new(IrohTransfer::new(node.clone()));

        let sessions = Arc::new(
            SessionManager::new(devices.to_vec(), transfer.clone())
                .with_telemetry(probe.clone())
                .with_custom_ops(custom_ops.clone()),
        );
        Self {
            node,
            sessions,
            transfer,
            authorizer,
        }
    }

    /// Drive an accepted session stream through the shared [`drive_session`] pump.
    ///
    /// The Iroh-specific parts are just the authenticated peer identity (`remote_id`, checked by the
    /// application's [`PeerAuthorizer`]) and this server's own id, echoed to the client.
    async fn handle_session(
        sessions: Arc<SessionManager<B, IrohTransfer<B>>>,
        authorizer: Arc<dyn PeerAuthorizer>,
        server_id: EndpointId,
        remote_id: EndpointId,
        send: SendStream,
        recv: RecvStream,
    ) -> Result<(), String> {
        drive_session(
            recv,
            send,
            sessions,
            Some(PeerId::Iroh(server_id)),
            |init| {
                authorizer.authorize(AuthorizationRequest {
                    peer: remote_id,
                    device_index: init.device_index,
                    credential: &init.authorization,
                })
            },
        )
        .await
    }
}

/// A backend-erased Burn Remote protocol handler.
///
/// The dispatch layer resolves a `Device` to a concrete backend and builds an
/// [`IrohRemoteProtocol`]; this wraps it as a single non-generic type, so an application can
/// register Burn on its own Iroh [`Router`] without naming a backend. Hand it directly to
/// [`RouterBuilder::accept`] under [`BURN_REMOTE_ALPN`].
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
        let remote_id = connection.remote_id();
        self.node.remember_connection(connection.clone()).await;
        loop {
            let Some((kind, send, recv)) = RemoteNode::accept_stream(&connection)
                .await
                .map_err(user_error)?
            else {
                return Ok(());
            };

            match kind {
                StreamKind::Session => {
                    let sessions = self.sessions.clone();
                    let authorizer = self.authorizer.clone();
                    let server_id = self.node.id();
                    spawn_detached(async move {
                        if let Err(err) = Self::handle_session(
                            sessions, authorizer, server_id, remote_id, send, recv,
                        )
                        .await
                        {
                            log::warn!("Rejected or failed Iroh remote session: {err}");
                        }
                    });
                }
                StreamKind::TensorTransfer => {
                    let transfer = self.transfer.clone();
                    spawn_detached(async move {
                        if let Err(err) = transfer.handle_stream(remote_id, send, recv).await {
                            log::warn!("Iroh tensor-transfer stream failed: {err}");
                        }
                    });
                }
            }
        }
    }
}

fn user_error(reason: String) -> AcceptError {
    AcceptError::from_err(std::io::Error::other(reason))
}

/// Serve Burn Remote over Iroh until the process receives its shutdown signal.
///
/// Binds a server endpoint with the stable identity carried by `secret` and hosts `devices` as the
/// sole protocol on it. Reached through [`RemoteServerBuilder`](super::RemoteServerBuilder) (the
/// single turnkey entry point); use [`RemoteNode::protocol`] for composition with other protocols.
#[cfg(not(target_family = "wasm"))]
pub(crate) async fn start_iroh_async<B: BackendIr>(
    secret: crate::RemoteSecret,
    devices: Vec<Device<B>>,
    custom_ops: CustomOpRegistry<B>,
) {
    use iroh::endpoint::presets;

    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret.secret_key())
        .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
        .bind()
        .await
        .expect("Can bind the Burn Remote server endpoint");

    let probe = if crate::metrics::TelemetryLogger::enabled() {
        TelemetryProbe::new(crate::telemetry::CHANNEL_CAPACITY)
    } else {
        TelemetryProbe::disabled()
    };

    let protocol = IrohRemoteProtocol::new(
        endpoint.clone(),
        devices,
        Arc::new(AllowAll),
        probe,
        custom_ops,
    );

    let router = Router::builder(endpoint)
        .accept(BURN_REMOTE_ALPN, protocol)
        .spawn();

    os_shutdown_signal().await;
    if let Err(err) = router.shutdown().await {
        log::warn!("Burn Remote Iroh router shutdown failed: {err}");
    }
}
