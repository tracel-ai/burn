//! Iroh protocol handler for Burn Remote compute and tensor-transfer streams.

use std::{fmt, sync::Arc};

use burn_ir::BackendIr;
use iroh::{
    endpoint::Connection,
    protocol::{AcceptError, DynProtocolHandler, ProtocolHandler},
};
use tokio_util::sync::CancellationToken;

use crate::{
    Credential, PeerId,
    server::{
        AuthorizationRequest, ClientId, PeerAuthorizer, SessionSetup, pump::drive_session,
        session::SessionManager, spawn::spawn_detached,
    },
    transport::message::MessageLimit,
};

use super::{
    IrohTransfer,
    node::{RemoteNode, StreamKind},
};

/// Serves Burn Remote's compute and tensor-transfer streams on an Iroh endpoint.
pub(crate) struct IrohRemoteProtocol<B: BackendIr> {
    node: RemoteNode,
    sessions: Arc<SessionManager<B, IrohTransfer<B>>>,
    transfer: Arc<IrohTransfer<B>>,
    authorizer: Arc<dyn PeerAuthorizer>,
    message_limit: MessageLimit,
    shutdown: CancellationToken,
}

/// A router dropped without `shutdown` drops its handlers, and the sessions they spawned would
/// otherwise outlive it.
impl<B: BackendIr> Drop for IrohRemoteProtocol<B> {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

impl<B: BackendIr> fmt::Debug for IrohRemoteProtocol<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IrohRemoteProtocol")
            .field("endpoint_id", &self.node.id())
            .finish_non_exhaustive()
    }
}

impl<B: BackendIr> IrohRemoteProtocol<B> {
    pub(crate) fn new(node: RemoteNode, setup: SessionSetup<B>) -> Self {
        let transfer = Arc::new(IrohTransfer::new(node.clone(), setup.message_limit));
        Self {
            sessions: Arc::new(setup.manager(transfer.clone())),
            node,
            transfer,
            authorizer: setup.authorizer,
            message_limit: setup.message_limit,
            shutdown: setup.shutdown,
        }
    }
}

/// Burn Remote's handler for an application's own Iroh router, from `into_protocol`. Register it
/// under [`BURN_REMOTE_ALPN`](crate::BURN_REMOTE_ALPN); shutting the router down ends its
/// sessions.
pub struct RemoteProtocol(Box<dyn DynProtocolHandler>);

impl RemoteProtocol {
    pub(crate) fn new(handler: impl ProtocolHandler) -> Self {
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
        let client_id = connection.remote_id();
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
                    let message_limit = self.message_limit;
                    let shutdown = self.shutdown.clone();
                    let server_id = self.node.id();
                    spawn_detached(async move {
                        let served = drive_session(
                            recv,
                            send,
                            sessions,
                            Some(PeerId::Iroh(server_id)),
                            &shutdown,
                            message_limit,
                            |init| {
                                authorizer.authorize(AuthorizationRequest {
                                    client: ClientId::Iroh(client_id),
                                    device_index: init.device_index,
                                    credential: &Credential::from(init.authorization.as_slice()),
                                })
                            },
                        )
                        .await;
                        if let Err(err) = served {
                            log::warn!("Rejected or failed Iroh remote session: {err}");
                        }
                    });
                }
                StreamKind::TensorTransfer => {
                    let transfer = self.transfer.clone();
                    spawn_detached(async move {
                        if let Err(err) = transfer.handle_stream(client_id, send, recv).await {
                            log::warn!("Iroh tensor-transfer stream failed: {err}");
                        }
                    });
                }
            }
        }
    }

    async fn shutdown(&self) {
        self.shutdown.cancel();
    }
}

fn user_error(reason: String) -> AcceptError {
    AcceptError::from_err(std::io::Error::other(reason))
}
