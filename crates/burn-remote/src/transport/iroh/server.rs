use crate::server::spawn::os_shutdown_signal;
use crate::telemetry::TelemetryProbe;
use crate::transport::iroh::node::BURN_REMOTE_ALPN;
use crate::transport::iroh::protocol::IrohRemoteProtocol;
#[cfg(not(target_family = "wasm"))]
use burn_backend::tensor::Device;
#[cfg(not(target_family = "wasm"))]
use burn_ir::BackendIr;
#[cfg(not(target_family = "wasm"))]
use burn_router::CustomOpRegistry;
use iroh::protocol::Router;

/// Serve Burn Remote over Iroh until the process receives its shutdown signal.
///
/// Binds a server endpoint with the stable identity carried by `secret` and hosts `devices` as the
/// sole protocol on it. Reached through [`RemoteServerBuilder`](super::RemoteServerBuilder) (the
/// single turnkey entry point); use [`RemoteNode::protocol`] for composition with other protocols.
#[cfg(not(target_family = "wasm"))]
pub(crate) async fn start_iroh_async<B: BackendIr>(
    channel: crate::IrohChannel,
    devices: Vec<Device<B>>,
    custom_ops: CustomOpRegistry<B>,
) {
    let mut builder = channel
        .relays
        .endpoint_builder(channel.segmentation_offload)
        .secret_key(channel.secret.secret_key())
        .alpns(vec![BURN_REMOTE_ALPN.to_vec()]);
    if let Some(port) = channel.port {
        builder = builder
            .clear_ip_transports()
            .bind_addr(format!("0.0.0.0:{port}"))
            .and_then(|builder| builder.bind_addr(format!("[::]:{port}")))
            .expect("A port makes valid bind addresses");
    }
    let endpoint = builder
        .bind()
        .await
        .expect("Can bind the Burn Remote server endpoint");
    log::info!("Burn Remote serving over Iroh as {}", channel.id());

    let probe = if crate::metrics::TelemetryLogger::enabled() {
        TelemetryProbe::new(crate::telemetry::CHANNEL_CAPACITY)
    } else {
        TelemetryProbe::disabled()
    };

    let protocol = IrohRemoteProtocol::new(
        endpoint.clone(),
        devices,
        channel.authorizer,
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
