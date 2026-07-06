//! The turnkey WebSocket compute server.

use std::sync::Arc;

use burn_backend::tensor::Device;
use burn_ir::BackendIr;
use burn_router::CustomOpRegistry;
use tokio_util::sync::CancellationToken;

use burn_communication::{
    ProtocolServer,
    external_comm::{ExternalCommServer, ExternalCommService},
    websocket::{WebSocket, WsServer, WsServerChannel},
};

use super::transfer::WebSocketTransfer;
use crate::server::{pump::drive_session, session::SessionManager, spawn::os_shutdown_signal};

/// Serve a WebSocket compute node on the given port, until shutdown.
///
/// The session protocol is a single full-duplex `/session` socket per session (split into a sink +
/// source and driven by the shared [`drive_session`] pump); cross-server tensor transfers ride the
/// same server via [`route_external_comm`](ExternalCommServer::route_external_comm). Driven through
/// [`RemoteServerBuilder`](crate::server::RemoteServerBuilder) rather than called directly.
#[cfg(not(target_family = "wasm"))]
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
        .route("/session", {
            let sessions = sessions.clone();
            move |channel: WsServerChannel| {
                let sessions = sessions.clone();
                async move {
                    let (sink, source) = channel.split();
                    // WebSocket has no authenticated peer identity, so the server presents none
                    // (`peer_id: None`) and authorizes every session.
                    if let Err(err) =
                        drive_session(source, sink, sessions, None, |_init| Ok(())).await
                    {
                        log::warn!("WebSocket remote session failed: {err}");
                    }
                }
            }
        })
        .route_external_comm(external);

    if let Err(err) = server.serve(os_shutdown_signal()).await {
        log::error!("Burn Remote WebSocket server stopped: {err:?}");
    }
}
