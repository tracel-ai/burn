#![cfg(feature = "websocket")]

use std::sync::Arc;

use burn_backend::tensor::Device;
use burn_ir::BackendIr;
use burn_router::CustomOpRegistry;
use tokio_util::sync::CancellationToken;

use burn_communication::{
    ProtocolServer,
    external_comm::{ExternalCommServer, ExternalCommService},
    websocket::{WebSocket, WsServer},
};

use super::{
    service::{FetchHandler, SubmitHandler},
    session::SessionManager,
    spawn::os_shutdown_signal,
    transfer::WebSocketTransfer,
};

/// Serve a WebSocket compute node on the given port, until shutdown.
///
/// `custom_ops` holds the handlers used to execute
/// [`OperationIr::Custom`](burn_ir::OperationIr::Custom) ops (e.g. from a backend extension); pass
/// [`CustomOpRegistry::default`] when hosting none. Driven through
/// [`RemoteServerBuilder`](super::RemoteServerBuilder) rather than called directly.
pub(crate) async fn start_websocket_async<B: BackendIr>(
    devices: Vec<Device<B>>,
    port: u16,
    custom_ops: CustomOpRegistry<B>,
) {
    let cancel_token = CancellationToken::new();
    let external = Arc::new(ExternalCommService::<B, WebSocket>::new(cancel_token));
    let transfer = Arc::new(WebSocketTransfer {
        inner: external.clone(),
    });
    let probe = if crate::metrics::TelemetryLogger::enabled() {
        crate::telemetry::TelemetryProbe::new(crate::telemetry::CHANNEL_CAPACITY)
    } else {
        crate::telemetry::TelemetryProbe::disabled()
    };
    let sessions = Arc::new(
        SessionManager::new(devices, transfer)
            .with_custom_ops(custom_ops)
            .with_telemetry(probe),
    );

    let server = WsServer::new(port)
        .route("/fetch", {
            let sessions = sessions.clone();
            move |stream| FetchHandler::new(sessions, stream).run()
        })
        .route("/submit", {
            let sessions = sessions.clone();
            move |stream| SubmitHandler::new(sessions, stream).run()
        })
        .route_external_comm(external);

    if let Err(err) = server.serve(os_shutdown_signal()).await {
        log::error!("Burn Remote WebSocket server stopped: {err:?}");
    }
}
