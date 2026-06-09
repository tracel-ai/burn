use burn_communication::{
    Protocol, ProtocolServer,
    external_comm::{ExternalCommServer, ExternalCommService},
    util::os_shutdown_signal,
    websocket::{WebSocket, WsServer},
};
use std::{marker::PhantomData, sync::Arc};
use tokio_util::sync::CancellationToken;

use burn_backend::tensor::Device;
use burn_ir::BackendIr;

use super::service::{FetchHandler, SubmitHandler};
use super::session::SessionManager;

/// HTTP-style server for the burn-remote protocol.
///
/// Each connection is a thin IO loop: [`FetchHandler`] answers the `/fetch` stream's init
/// handshake and drains the session's result queue, while [`SubmitHandler`] decodes the
/// `/submit` stream's message batches and forwards each task to the session's worker. The
/// tasks actually run on that per-session worker thread (see
/// [`SessionWorker`](super::worker::SessionWorker)), which holds the session's runner and
/// processes its tasks in FIFO order — so a blocking op (e.g. an all-reduce barrier) parks
/// only that session's worker rather than a shared runtime thread. The [`SessionManager`] owns
/// the per-session state behind the [`SubmitService`](super::service::SubmitService) /
/// [`FetchService`](super::service::FetchService) traits the handlers depend on.
pub struct RemoteServer<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    _b: PhantomData<B>,
    _p: PhantomData<P>,
}

impl<B, P> RemoteServer<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Start the server hosting the given devices.
    ///
    /// `devices` is indexed by the device index the client selects at session init;
    /// `devices[0]` is the default device. Must be non-empty.
    pub async fn start(devices: Vec<Device<B>>, server: P::Server) {
        let cancel_token = CancellationToken::new();
        let external_comm = Arc::new(ExternalCommService::<B, P>::new(cancel_token));
        let session_manager = Arc::new(SessionManager::<B, P>::new(devices, external_comm.clone()));

        let _server = server
            .route("/fetch", {
                let session_manager = session_manager.clone();
                move |stream| FetchHandler::new(session_manager, stream).run()
            })
            .route("/submit", {
                let session_manager = session_manager.clone();
                move |stream| SubmitHandler::new(session_manager, stream).run()
            })
            .route_external_comm(external_comm)
            .serve(os_shutdown_signal())
            .await;
    }
}

/// Start the server on the given port, hosting the given [devices](Device).
///
/// `devices` is indexed by the device index the client selects; `devices[0]` is the default.
pub async fn start_websocket_async<B: BackendIr>(devices: Vec<Device<B>>, port: u16) {
    let server = WsServer::new(port);
    RemoteServer::<B, WebSocket>::start(devices, server).await;
}

#[tokio::main]
/// Start the server on the given port, hosting the given [devices](Device).
pub async fn start_websocket<B: BackendIr>(devices: Vec<Device<B>>, port: u16) {
    start_websocket_async::<B>(devices, port).await;
}
