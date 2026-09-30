use std::sync::Arc;

use iroh::EndpointId;

use super::{
    protocol::{AllowAll, PeerAuthorizer},
    relays::IrohRelays,
    secret::RemoteSecret,
};

/// How a server serves over Iroh: its identity, its relays, the port it binds, and whom it serves.
/// Built with [`IrohChannelBuilder`].
#[derive(Clone)]
pub struct IrohChannel {
    secret: RemoteSecret,
    relays: IrohRelays,
    port: Option<u16>,
    authorizer: Arc<dyn PeerAuthorizer>,
}

impl IrohChannel {
    /// The id clients dial.
    pub fn id(&self) -> EndpointId {
        self.secret.id()
    }
}

#[cfg(not(target_family = "wasm"))]
mod serve {
    use burn_backend::tensor::Device;
    use burn_ir::BackendIr;
    use burn_router::CustomOpRegistry;
    use iroh::{endpoint::BindOpts, protocol::Router};

    use super::IrohChannel;
    use crate::{
        server::spawn::os_shutdown_signal,
        telemetry::TelemetryProbe,
        transport::iroh::{node::BURN_REMOTE_ALPN, protocol::IrohRemoteProtocol},
    };

    impl IrohChannel {
        /// Serve `devices` until the process receives its shutdown signal.
        pub(crate) async fn serve<B: BackendIr>(
            self,
            devices: Vec<Device<B>>,
            custom_ops: CustomOpRegistry<B>,
        ) {
            let mut builder = self
                .relays
                .endpoint_builder()
                .secret_key(self.secret.secret_key())
                .alpns(vec![BURN_REMOTE_ALPN.to_vec()]);
            if let Some(port) = self.port {
                // Optional like Iroh's own IPv6 bind, so a host without IPv6 still serves on IPv4.
                let ipv6 = BindOpts::default().set_is_required(false);
                builder = builder
                    .clear_ip_transports()
                    .bind_addr(format!("0.0.0.0:{port}"))
                    .and_then(|builder| builder.bind_addr_with_opts(format!("[::]:{port}"), ipv6))
                    .expect("A port makes valid bind addresses");
            }
            let endpoint = builder
                .bind()
                .await
                .expect("Can bind the Burn Remote server endpoint");
            log::info!("Burn Remote serving over Iroh as {}", self.id());

            let probe = if crate::metrics::TelemetryLogger::enabled() {
                TelemetryProbe::new(crate::telemetry::CHANNEL_CAPACITY)
            } else {
                TelemetryProbe::disabled()
            };
            let protocol = IrohRemoteProtocol::new(
                endpoint.clone(),
                devices,
                self.authorizer,
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
    }
}

impl core::fmt::Debug for IrohChannel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The public identity only, never the secret key material.
        f.debug_struct("IrohChannel")
            .field("id", &self.id())
            .field("relays", &self.relays)
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

/// Builds an [`IrohChannel`]. Unless set otherwise, it serves every peer through n0's public
/// relays, on a port the OS picks.
#[derive(Clone, Debug)]
pub struct IrohChannelBuilder {
    channel: IrohChannel,
}

impl IrohChannelBuilder {
    /// A channel serving as `secret`'s id.
    pub fn new(secret: RemoteSecret) -> Self {
        Self {
            channel: IrohChannel {
                secret,
                relays: IrohRelays::default(),
                port: None,
                authorizer: Arc::new(AllowAll),
            },
        }
    }

    /// The relays to reach clients through.
    pub fn relays(mut self, relays: IrohRelays) -> Self {
        self.channel.relays = relays;
        self
    }

    /// Bind UDP `port` on every IPv4 and IPv6 interface, so clients can dial it directly. Needed
    /// when relays are disabled, since nothing else tells clients where the server is.
    pub fn port(mut self, port: u16) -> Self {
        self.channel.port = Some(port);
        self
    }

    /// Serve only the sessions `authorizer` accepts.
    pub fn authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.channel.authorizer = Arc::new(authorizer);
        self
    }

    /// The channel.
    pub fn build(self) -> IrohChannel {
        self.channel
    }
}
