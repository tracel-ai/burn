//! Serving over Iroh.

use core::fmt;

use burn_ir::BackendIr;
use iroh::{Endpoint, EndpointId, endpoint::BindOpts, protocol::Router};

use super::{
    identity::IrohIdentity,
    node::{BURN_REMOTE_ALPN, RemoteNode},
    protocol::IrohRemoteProtocol,
    relays::IrohRelays,
};
use crate::server::{ServeError, SessionSetup};

/// How a server accepts clients over Iroh: the identity they dial, the relays they reach it
/// through, and the UDP port it binds.
#[derive(Clone)]
pub struct IrohTransport {
    // Boxed because a key is 224 bytes, which would bloat `Transport` for every transport.
    identity: Box<IrohIdentity>,
    relays: IrohRelays,
    port: Option<u16>,
}

impl IrohTransport {
    /// Serve as `identity`'s [`id`](Self::id). Reached through n0's public relays, on a port the
    /// OS picks, unless set otherwise.
    pub fn new(identity: IrohIdentity) -> Self {
        Self {
            identity: Box::new(identity),
            relays: IrohRelays::default(),
            port: None,
        }
    }

    /// The relays clients reach the server through.
    pub fn with_relays(mut self, relays: IrohRelays) -> Self {
        self.relays = relays;
        self
    }

    /// Bind UDP `port` on IPv4, and on IPv6 unless the host has none or the port is taken there,
    /// so clients can dial it directly. Needed when relays are disabled, since nothing else tells
    /// clients where the server is.
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// The id clients dial.
    pub fn id(&self) -> EndpointId {
        self.identity.id()
    }

    /// Bind the endpoint, ready to serve on.
    pub(crate) async fn bind(self) -> Result<IrohListener, ServeError> {
        let endpoint = self.bind_endpoint().await?;
        log::info!(
            "Burn Remote serving over Iroh as {} on {:?}",
            self.id(),
            endpoint.bound_sockets()
        );
        if self.relays == IrohRelays::Disabled && self.port.is_none() {
            log::warn!("Relays disabled without a port: clients can only dial the ports above");
        }
        Ok(IrohListener {
            node: RemoteNode::from_endpoint(endpoint),
        })
    }

    async fn bind_endpoint(&self) -> Result<Endpoint, ServeError> {
        let mut builder = self
            .relays
            .endpoint_builder()
            .secret_key(self.identity.secret_key())
            .alpns(vec![BURN_REMOTE_ALPN.to_vec()]);
        if let Some(port) = self.port {
            // Optional like Iroh's own IPv6 bind, so a host without IPv6 still serves on IPv4.
            let ipv6 = BindOpts::default().set_is_required(false);
            builder = builder
                .clear_ip_transports()
                .bind_addr(format!("0.0.0.0:{port}"))
                .and_then(|builder| builder.bind_addr_with_opts(format!("[::]:{port}"), ipv6))
                .map_err(ServeError::bind)?;
        }
        builder.bind().await.map_err(ServeError::bind)
    }
}

impl fmt::Debug for IrohTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IrohTransport")
            .field("id", &self.id())
            .field("relays", &self.relays)
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

/// A bound Iroh endpoint, which accepts clients once served.
pub(crate) struct IrohListener {
    node: RemoteNode,
}

impl IrohListener {
    /// Serve until `setup`'s shutdown is cancelled, the endpoint closes, or the returned future is
    /// dropped.
    pub(crate) async fn serve<B: BackendIr>(
        self,
        setup: SessionSetup<B>,
    ) -> Result<(), ServeError> {
        let endpoint = self.node.endpoint().clone();
        let closed = endpoint.closed();
        let shutdown = setup.shutdown.clone();
        let router = ShutdownOnDrop(Some(
            Router::builder(endpoint)
                .accept(BURN_REMOTE_ALPN, IrohRemoteProtocol::new(self.node, setup))
                .spawn(),
        ));
        let served = tokio::select! {
            () = shutdown.cancelled() => Ok(()),
            () = closed => Err(ServeError::transport("the Iroh endpoint closed while serving")),
        };
        router.shutdown().await;
        served
    }
}

/// Shuts a router down, closing its connections so clients learn at once that the server is gone.
/// Dropped without [`shutdown`](Self::shutdown), as when the serving future is, it can only start
/// the shutdown in the background.
struct ShutdownOnDrop(Option<Router>);

impl ShutdownOnDrop {
    async fn shutdown(mut self) {
        if let Some(router) = self.0.take() {
            Self::close(router).await;
        }
    }

    async fn close(router: Router) {
        if let Err(err) = router.shutdown().await {
            log::warn!("Burn Remote Iroh router shutdown failed: {err}");
        }
    }
}

impl Drop for ShutdownOnDrop {
    fn drop(&mut self) {
        if let Some(router) = self.0.take()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            runtime.spawn(Self::close(router));
        }
    }
}
