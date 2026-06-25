use std::{fmt, sync::Arc};

use burn_backend::tensor::Device;
use burn_ir::BackendIr;
use burn_router::CustomOpRegistry;
use iroh::{
    EndpointId,
    endpoint::{Connection, RecvStream, SendStream},
    protocol::{AcceptError, DynProtocolHandler, ProtocolHandler, Router, RouterBuilder},
};

use crate::{
    BURN_REMOTE_ALPN, PeerId, RemoteNode,
    server::{
        service::{FetchService, SubmitService, parse_init_handshake},
        session::SessionManager,
        spawn::spawn_detached,
        transfer::IrohTransfer,
    },
    shared::{PROTOCOL_VERSION, RemoteMessage, SessionInfo, TaskResponse, TaskResponseContent},
    telemetry::TelemetryProbe,
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
struct AllowAll;

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
    devices: Vec<Device<B>>,
    sessions: Arc<SessionManager<B, IrohTransfer<B>>>,
    transfer: Arc<IrohTransfer<B>>,
    authorizer: Arc<dyn PeerAuthorizer>,
    probe: TelemetryProbe,
    custom_ops: CustomOpRegistry<B>,
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
    pub fn new(node: RemoteNode, devices: Vec<Device<B>>) -> Self {
        let transfer = Arc::new(IrohTransfer::new(node.clone()));
        let probe = TelemetryProbe::disabled();
        let custom_ops = CustomOpRegistry::default();
        let sessions = Arc::new(Self::build_sessions(
            &devices,
            &transfer,
            &probe,
            &custom_ops,
        ));
        Self {
            node,
            devices,
            sessions,
            transfer,
            authorizer: Arc::new(AllowAll),
            probe,
            custom_ops,
        }
    }

    /// Emit per-session telemetry into `probe` for live monitoring.
    pub fn with_telemetry(mut self, probe: TelemetryProbe) -> Self {
        self.probe = probe;
        self.rebuild_sessions();
        self
    }

    /// Register custom-op handlers, so a backend extension can host its
    /// [`OperationIr::Custom`](burn_ir::OperationIr::Custom) ops over Iroh.
    pub fn with_custom_ops(mut self, custom_ops: CustomOpRegistry<B>) -> Self {
        self.custom_ops = custom_ops;
        self.rebuild_sessions();
        self
    }

    /// Install an application authorization policy.
    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.authorizer = Arc::new(authorizer);
        self
    }

    /// Rebuild the session manager from the current telemetry and custom-op configuration. No
    /// session has been accepted yet, so this just swaps in a freshly configured manager.
    fn rebuild_sessions(&mut self) {
        self.sessions = Arc::new(Self::build_sessions(
            &self.devices,
            &self.transfer,
            &self.probe,
            &self.custom_ops,
        ));
    }

    fn build_sessions(
        devices: &[Device<B>],
        transfer: &Arc<IrohTransfer<B>>,
        probe: &TelemetryProbe,
        custom_ops: &CustomOpRegistry<B>,
    ) -> SessionManager<B, IrohTransfer<B>> {
        SessionManager::new(devices.to_vec(), transfer.clone())
            .with_telemetry(probe.clone())
            .with_custom_ops(custom_ops.clone())
    }

    /// Install a shared authorization policy. Used by the dispatch layer, which carries the policy
    /// as a trait object across the backend-erasure boundary.
    pub fn with_authorizer_arc(mut self, authorizer: Arc<dyn PeerAuthorizer>) -> Self {
        self.authorizer = authorizer;
        self
    }

    /// Pre-load this protocol onto a [`RouterBuilder`] for the node's endpoint, left unspawned so
    /// the caller can register other Iroh protocols before calling `.spawn()`.
    pub fn into_builder(self) -> RouterBuilder {
        let endpoint = self.node.endpoint().clone();
        Router::builder(endpoint).accept(BURN_REMOTE_ALPN, self)
    }

    /// Serve this protocol as the sole protocol on the node's endpoint.
    pub fn serve(self) -> Router {
        self.into_builder().spawn()
    }

    async fn handle_session(
        sessions: Arc<SessionManager<B, IrohTransfer<B>>>,
        authorizer: Arc<dyn PeerAuthorizer>,
        server_id: EndpointId,
        remote_id: EndpointId,
        mut send: SendStream,
        mut recv: RecvStream,
    ) -> Result<(), String> {
        let handshake = crate::node::recv_frame(&mut recv)
            .await?
            .ok_or_else(|| "Session stream closed before initialization".to_string())?;
        let init = parse_init_handshake(&handshake)?;

        authorizer.authorize(AuthorizationRequest {
            peer: remote_id,
            device_index: init.device_index,
            credential: &init.authorization,
        })?;

        let task_sender = sessions
            .session_task_sender(init.session_id, init.device_index)
            .await;
        let mut responses = sessions
            .take_response_receiver(init.session_id, init.device_index)
            .await?;

        let info = TaskResponse {
            id: 0,
            content: TaskResponseContent::Init(SessionInfo {
                version: PROTOCOL_VERSION,
                settings: sessions.device_settings(init.device_index),
                device_count: sessions.device_count(),
                peer_id: Some(PeerId::Iroh(server_id)),
            }),
        };
        let info = rmp_serde::to_vec(&info)
            .map_err(|err| format!("Failed to encode session handshake response: {err}"))?;
        crate::node::send_frame(&mut send, &info).await?;

        let (writer_done, writer_result) = tokio::sync::oneshot::channel();
        spawn_detached(async move {
            let result = async {
                while let Some(response) = responses.recv().await {
                    let bytes = rmp_serde::to_vec(&response)
                        .map_err(|err| format!("Failed to encode task response: {err}"))?;
                    crate::node::send_frame(&mut send, &bytes).await?;
                }
                send.finish()
                    .map_err(|err| format!("Failed to finish session response stream: {err}"))
            }
            .await;
            let _ = writer_done.send(result);
        });

        let result = loop {
            let Some(frame) = crate::node::recv_frame(&mut recv).await? else {
                break Ok(());
            };
            let messages: Vec<RemoteMessage> = rmp_serde::from_slice(&frame)
                .map_err(|err| format!("Invalid remote task batch: {err}"))?;
            let mut close = false;
            let mut protocol_error = None;
            for message in messages {
                match message {
                    RemoteMessage::Task(task) => {
                        task_sender
                            .send(task)
                            .await
                            .map_err(|_| "Session worker stopped".to_string())?;
                    }
                    RemoteMessage::Close(id) if id == init.session_id => {
                        close = true;
                        break;
                    }
                    RemoteMessage::Close(id) => {
                        protocol_error = Some(format!(
                            "Session {} attempted to close unrelated session {id}",
                            init.session_id
                        ));
                        break;
                    }
                    RemoteMessage::Init(_) => {
                        protocol_error =
                            Some("A session stream cannot be initialized twice".into());
                        break;
                    }
                }
            }
            if let Some(err) = protocol_error {
                break Err(err);
            }
            if close {
                break Ok(());
            }
        };

        drop(task_sender);
        sessions.close(init.session_id).await;
        match writer_result.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => log::warn!("Iroh response writer failed: {err}"),
            Err(_) => log::warn!("Iroh response writer task stopped before finishing"),
        }
        result
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
                crate::node::StreamKind::Session => {
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
                crate::node::StreamKind::TensorTransfer => {
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

impl RemoteNode {
    /// Build a composable Iroh protocol handler hosting `devices`.
    ///
    /// This is the low-level, backend-generic primitive. Most callers go through the dispatch
    /// surface in `burn::server`, which selects the backend from a `Device` and erases it behind
    /// [`RemoteProtocol`]; reach for this directly only when hosting a custom `BackendIr` that
    /// isn't part of the dispatch backend. The returned handler is a builder: chain
    /// [`with_authorizer`](IrohRemoteProtocol::with_authorizer) /
    /// [`with_telemetry`](IrohRemoteProtocol::with_telemetry), then register it on your own
    /// [`Router`] or call [`serve`](IrohRemoteProtocol::serve) to host it alone.
    pub fn protocol<B: BackendIr>(&self, devices: Vec<Device<B>>) -> IrohRemoteProtocol<B> {
        IrohRemoteProtocol::new(self.clone(), devices)
    }
}

/// Serve Burn Remote over Iroh until the process receives its shutdown signal.
///
/// Binds a server endpoint with the stable identity carried by `secret` and hosts `devices` as the
/// sole protocol on it.
#[cfg(not(target_family = "wasm"))]
pub async fn start_iroh_async<B: BackendIr>(
    secret: crate::RemoteSecret,
    devices: Vec<Device<B>>,
    custom_ops: CustomOpRegistry<B>,
) {
    let node = RemoteNode::bind_with_secret(&secret)
        .await
        .expect("Can bind the Burn Remote server endpoint");
    let router = node
        .protocol::<B>(devices)
        .with_custom_ops(custom_ops)
        .serve();
    os_shutdown_signal().await;
    if let Err(err) = router.shutdown().await {
        log::warn!("Burn Remote Iroh router shutdown failed: {err}");
    }
}

/// Serve Burn Remote over Iroh, blocking the current thread.
#[cfg(not(target_family = "wasm"))]
#[tokio::main]
pub async fn start_iroh<B: BackendIr>(
    secret: crate::RemoteSecret,
    devices: Vec<Device<B>>,
    custom_ops: CustomOpRegistry<B>,
) {
    start_iroh_async::<B>(secret, devices, custom_ops).await;
}

/// Resolve when the process is asked to stop (Ctrl+C, or `SIGTERM` on Unix).
#[cfg(not(target_family = "wasm"))]
async fn os_shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
