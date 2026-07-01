use crate::server::spawn::os_shutdown_signal;
use crate::telemetry::TelemetryProbe;
use crate::transport::iroh::node::BURN_REMOTE_ALPN;
use crate::transport::iroh::protocol::{AllowAll, IrohRemoteProtocol};
#[cfg(not(target_family = "wasm"))]
use burn_backend::tensor::Device;
#[cfg(not(target_family = "wasm"))]
use burn_ir::BackendIr;
#[cfg(not(target_family = "wasm"))]
use burn_router::CustomOpRegistry;
use iroh::{Endpoint, endpoint::presets, protocol::Router};
use std::sync::Arc;

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
