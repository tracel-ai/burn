//! Serving this machine's devices to remote clients.
//!
//! ```rust,ignore
//! let identity = IrohIdentity::load_or_create("server.key")?;
//! let transport = IrohTransport::new(identity);
//! println!("server id: {}", transport.id());
//!
//! RemoteServer::new([Device::cuda(0)])
//!     .with_authorizer(TokenAuthorizer::new(token)?)
//!     .serve(transport)?;
//! ```
//!
//! Iroh reaches a server across any network, authenticated and encrypted. WebSocket, with the
//! `remote-websocket` feature, is the simplest setup on a trusted network. A backend outside
//! Burn's own serves through `burn_remote::server::BackendServer`.

#[cfg(not(target_family = "wasm"))]
use core::future::Future;

#[cfg(all(not(target_family = "wasm"), feature = "remote-websocket"))]
pub use burn_dispatch::__remote::server::WebSocketTransport;
#[cfg(not(target_family = "wasm"))]
pub use burn_dispatch::__remote::server::{IrohTransport, Transport};
pub use burn_dispatch::__remote::{
    BURN_REMOTE_ALPN, Credential, Endpoint, InvalidRelays, IrohRelays, RelayUrl,
    ir::{CustomOpIr, HandleContainer},
    server::{
        AllowAll, AuthorizationRequest, ClientId, CustomOpRegistry, EmptyToken, IrohIdentity,
        PeerAuthorizer, RemoteProtocol, ServeError, ServerLogging, TokenAuthorizer,
    },
    telemetry,
};
use burn_dispatch::__remote::{ir::BackendIr, server::ServerSettings};

use crate::Device;
use telemetry::TelemetryProbe;

/// A server running tensor operations on its devices for remote clients.
///
/// The devices pick the backend, and must all belong to one: every CubeCL runtime is one backend,
/// so CUDA and wgpu devices can be served together, but not with Flex or NdArray ones. Autodiff is
/// stripped, since the autodiff graph is the client's.
#[derive(Clone)]
pub struct RemoteServer {
    devices: Vec<Device>,
    settings: ServerSettings,
}

impl RemoteServer {
    /// Host `devices`. A client picks one by its position in this list.
    pub fn new(devices: impl IntoIterator<Item = Device>) -> Self {
        Self {
            devices: devices.into_iter().collect(),
            settings: ServerSettings::default(),
        }
    }

    /// Open only the sessions `authorizer` accepts. Every session is opened unless set.
    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.settings = self.settings.with_authorizer(authorizer);
        self
    }

    /// Report every session's activity to `probe`.
    pub fn with_telemetry(mut self, probe: TelemetryProbe) -> Self {
        self.settings = self.settings.with_telemetry(probe);
        self
    }

    /// Run `handler` for the custom operation `id` on backend `B`, which must be the devices'
    /// backend. Every CubeCL runtime is the backend `Cube`, so a handler for one branches on the
    /// device's runtime.
    pub fn with_custom_op<B: BackendIr, F>(mut self, id: &str, handler: F) -> Self
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        self.settings = self.settings.with_custom_op::<B, F>(id, handler);
        self
    }

    /// Replace backend `B`'s custom operations with `registry`.
    pub fn with_custom_ops<B: BackendIr>(mut self, registry: CustomOpRegistry<B>) -> Self {
        self.settings = self.settings.with_custom_ops(registry);
        self
    }

    /// Serve on `transport`, blocking the calling thread until the process receives Ctrl+C or
    /// `SIGTERM`. Returning cancels every session; Iroh connections are closed before it returns.
    ///
    /// Installs [`ServerLogging`] first, and the signal handlers only once the transport has
    /// bound, so a server that cannot start leaves Ctrl+C as it was. The server runs on Burn's own
    /// runtime, so this can be called from any thread, inside an async runtime or not.
    #[cfg(not(target_family = "wasm"))]
    pub fn serve(&self, transport: impl Into<Transport>) -> Result<(), ServeError> {
        burn_dispatch::remote_server::serve(
            self.dispatch_devices(),
            self.settings.clone(),
            transport.into(),
        )
    }

    /// Serve on `transport` until the returned future is dropped, which also ends the live
    /// sessions. An Iroh port is free again shortly after, once its connections have closed.
    ///
    /// Requires a Tokio runtime. Installs no logging and no signal handlers: those belong to the
    /// application.
    #[cfg(not(target_family = "wasm"))]
    pub fn serve_async<T: Into<Transport>>(
        &self,
        transport: T,
    ) -> impl Future<Output = Result<(), ServeError>> + Send + 'static + use<T> {
        burn_dispatch::remote_server::serve_async(
            self.dispatch_devices(),
            self.settings.clone(),
            transport.into(),
        )
    }

    /// Burn Remote's handler for the application's own Iroh router on `endpoint`, to register
    /// under [`BURN_REMOTE_ALPN`] beside its other protocols. Shutting the router down ends the
    /// sessions.
    pub fn into_protocol(self, endpoint: &Endpoint) -> Result<RemoteProtocol, ServeError> {
        let devices = self
            .devices
            .into_iter()
            .map(Device::into_dispatch)
            .collect();
        burn_dispatch::remote_server::into_protocol(devices, self.settings, endpoint)
    }

    #[cfg(not(target_family = "wasm"))]
    fn dispatch_devices(&self) -> Vec<burn_dispatch::DispatchDevice> {
        self.devices
            .iter()
            .cloned()
            .map(Device::into_dispatch)
            .collect()
    }
}
