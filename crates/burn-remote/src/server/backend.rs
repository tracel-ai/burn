#[cfg(not(target_family = "wasm"))]
use core::future::Future;

use burn_backend::tensor::Device;
use burn_ir::{BackendIr, CustomOpIr, HandleContainer};
use burn_router::CustomOpRegistry;
use tokio_util::sync::CancellationToken;

use super::{PeerAuthorizer, ServeError, ServerSettings};
#[cfg(not(target_family = "wasm"))]
use super::{ServerLogging, Transport, spawn::os_shutdown_signal};
#[cfg(not(target_family = "wasm"))]
use crate::runtime;
use crate::telemetry::TelemetryProbe;
#[cfg(feature = "iroh")]
use crate::{
    Endpoint,
    server::RemoteProtocol,
    transport::iroh::{node::RemoteNode, protocol::IrohRemoteProtocol},
};

/// A server hosting devices of backend `B`.
///
/// `burn::server::RemoteServer` serves Burn's own backends, picking the backend from its devices;
/// this serves any [`BackendIr`] backend, one outside Burn included. Custom operations are typed by
/// `B`, since their handlers call into `B`'s primitives.
///
/// ```rust,ignore
/// BackendServer::<MyBackend>::new([MyDevice::default()])
///     .with_custom_op("fused_matmul_add_relu", |handles, ir, _device| {
///         let ([lhs, rhs, bias], [out]) = ir.as_fixed::<3, 1>();
///         let lhs = handles.get_float_tensor::<MyBackend>(lhs);
///         let rhs = handles.get_float_tensor::<MyBackend>(rhs);
///         let bias = handles.get_float_tensor::<MyBackend>(bias);
///         let result = <MyBackend as MyExt>::fused_matmul_add_relu(lhs, rhs, bias);
///         handles.register_float_tensor::<MyBackend>(&out.id, result);
///     })
///     .serve(WebSocketTransport::new(3000))?;
/// ```
pub struct BackendServer<B: BackendIr> {
    devices: Vec<Device<B>>,
    settings: ServerSettings,
}

impl<B: BackendIr> BackendServer<B> {
    /// Host `devices`. A client picks one by its position in this list.
    pub fn new(devices: impl IntoIterator<Item = Device<B>>) -> Self {
        Self {
            devices: devices.into_iter().collect(),
            settings: ServerSettings::default(),
        }
    }

    #[doc(hidden)]
    pub fn with_settings(mut self, settings: ServerSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Open only the sessions `authorizer` accepts. Every session is opened unless set.
    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.settings = self.settings.with_authorizer(authorizer);
        self
    }

    /// Report every session's activity to `probe`. Unless set, sessions are logged at the remote
    /// logger level of Burn's config.
    pub fn with_telemetry(mut self, probe: TelemetryProbe) -> Self {
        self.settings = self.settings.with_telemetry(probe);
        self
    }

    /// Run `handler` for the [custom operation](burn_ir::OperationIr::Custom) `id`, the one the
    /// client puts in its [`CustomOpIr`]. Registering an id again replaces its handler.
    pub fn with_custom_op<F>(mut self, id: &str, handler: F) -> Self
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        self.settings = self.settings.with_custom_op::<B, F>(id, handler);
        self
    }

    /// Replace the custom operations with `registry`.
    pub fn with_custom_ops(mut self, registry: CustomOpRegistry<B>) -> Self {
        self.settings = self.settings.with_custom_ops(registry);
        self
    }

    /// Serve on `transport`, blocking the calling thread until the process receives Ctrl+C or
    /// `SIGTERM`.
    ///
    /// Installs [`ServerLogging`] and the signal handlers. The server runs on Burn's own runtime,
    /// so this can be called from any thread, inside an async runtime or not.
    ///
    /// # Errors
    ///
    /// See [`ServeError`].
    #[cfg(not(target_family = "wasm"))]
    pub fn serve(&self, transport: impl Into<Transport>) -> Result<(), ServeError> {
        ServerLogging::install();
        let serving = self.serve_async(transport);
        runtime::wait(move || {
            runtime::blocking_runtime().handle().block_on(async move {
                tokio::select! {
                    served = serving => served,
                    stopped = os_shutdown_signal() => stopped,
                }
            })
        })
    }

    /// Serve on `transport` until the returned future is dropped, which also ends the live
    /// sessions.
    ///
    /// Requires a Tokio runtime. Installs no logging and no signal handlers: those belong to the
    /// application.
    ///
    /// # Errors
    ///
    /// See [`ServeError`].
    #[cfg(not(target_family = "wasm"))]
    pub fn serve_async<T: Into<Transport>>(
        &self,
        transport: T,
    ) -> impl Future<Output = Result<(), ServeError>> + Send + 'static + use<B, T> {
        let transport = transport.into();
        let devices = self.devices.clone();
        let settings = self.settings.clone();
        async move {
            let shutdown = CancellationToken::new();
            let _ends_sessions = shutdown.clone().drop_guard();
            transport
                .serve(settings.sessions_of::<B>(devices, shutdown)?)
                .await
        }
    }

    /// Burn Remote's handler for the application's own Iroh router on `endpoint`, to register
    /// under [`BURN_REMOTE_ALPN`](crate::BURN_REMOTE_ALPN) beside its other protocols. Shutting
    /// the router down ends the sessions.
    ///
    /// # Errors
    ///
    /// See [`ServeError`].
    #[cfg(feature = "iroh")]
    pub fn into_protocol(self, endpoint: &Endpoint) -> Result<RemoteProtocol, ServeError> {
        let node = RemoteNode::for_endpoint(endpoint)
            .map_err(|reason| ServeError::InvalidEndpoint { reason })?;
        let setup = self
            .settings
            .sessions_of::<B>(self.devices, CancellationToken::new())?;
        Ok(RemoteProtocol::new(IrohRemoteProtocol::new(node, setup)))
    }
}
