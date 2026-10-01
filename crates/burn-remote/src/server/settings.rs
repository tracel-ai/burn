use std::{
    any::{Any, TypeId, type_name},
    collections::HashMap,
    sync::Arc,
};

use burn_backend::tensor::Device;
use burn_ir::{BackendIr, CustomOpIr, HandleContainer};
use burn_router::CustomOpRegistry;
use tokio_util::sync::CancellationToken;

use super::{
    AllowAll, PeerAuthorizer, ServeError, session::SessionManager, transfer::TensorTransfer,
};
use crate::telemetry::{CHANNEL_CAPACITY, TelemetryProbe};

/// What a server applies to every session, whatever its transport: its authorizer, its telemetry
/// and its custom operations.
///
/// Custom operations are kept per backend and checked against the devices' backend when the
/// server starts, so a server that learns its backend from its devices can hold them.
#[doc(hidden)]
#[derive(Clone)]
pub struct ServerSettings {
    authorizer: Arc<dyn PeerAuthorizer>,
    telemetry: Option<TelemetryProbe>,
    custom_ops: HashMap<TypeId, BackendOps>,
}

/// One backend's custom operations: a `CustomOpRegistry` of the backend it names.
#[derive(Clone)]
struct BackendOps {
    backend: &'static str,
    registry: Arc<dyn Any + Send + Sync>,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            authorizer: Arc::new(AllowAll),
            telemetry: None,
            custom_ops: HashMap::new(),
        }
    }
}

impl ServerSettings {
    /// Open only the sessions `authorizer` accepts.
    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.authorizer = Arc::new(authorizer);
        self
    }

    /// Report every session's activity to `probe`.
    pub fn with_telemetry(mut self, probe: TelemetryProbe) -> Self {
        self.telemetry = Some(probe);
        self
    }

    /// Run `handler` for the custom operation `id` on backend `B`.
    pub fn with_custom_op<B: BackendIr, F>(self, id: &str, handler: F) -> Self
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        let mut registry = self.custom_ops_of::<B>();
        registry.register(id, handler);
        self.with_custom_ops(registry)
    }

    /// Replace backend `B`'s custom operations with `registry`.
    pub fn with_custom_ops<B: BackendIr>(mut self, registry: CustomOpRegistry<B>) -> Self {
        self.custom_ops.insert(
            TypeId::of::<B>(),
            BackendOps {
                backend: type_name::<B>(),
                registry: Arc::new(registry),
            },
        );
        self
    }

    fn custom_ops_of<B: BackendIr>(&self) -> CustomOpRegistry<B> {
        self.custom_ops
            .get(&TypeId::of::<B>())
            .and_then(|ops| ops.registry.downcast_ref::<CustomOpRegistry<B>>())
            .cloned()
            .unwrap_or_default()
    }

    /// Everything a session of `devices` is set up with, checking that the custom operations are
    /// for their backend.
    pub(crate) fn sessions_of<B: BackendIr>(
        &self,
        devices: Vec<Device<B>>,
        shutdown: CancellationToken,
    ) -> Result<SessionSetup<B>, ServeError> {
        if devices.is_empty() {
            return Err(ServeError::NoDevices);
        }
        if let Some(other) = self
            .custom_ops
            .iter()
            .find_map(|(backend, ops)| (*backend != TypeId::of::<B>()).then_some(ops))
        {
            return Err(ServeError::CustomOpBackend {
                backend: other.backend,
            });
        }
        // Readbacks on a server's async runtime must not park a blocking device-to-host copy on
        // an executor worker; this makes them eager, for the whole process.
        burn_std::set_runtime_kind(burn_std::RuntimeKind::Async);
        Ok(SessionSetup {
            devices,
            custom_ops: self.custom_ops_of::<B>(),
            telemetry: self.telemetry.clone().unwrap_or_else(|| {
                if crate::metrics::TelemetryLogger::enabled() {
                    TelemetryProbe::new(CHANNEL_CAPACITY)
                } else {
                    TelemetryProbe::disabled()
                }
            }),
            authorizer: self.authorizer.clone(),
            shutdown,
        })
    }
}

/// What every session of a starting server is set up with.
pub(crate) struct SessionSetup<B: BackendIr> {
    devices: Vec<Device<B>>,
    custom_ops: CustomOpRegistry<B>,
    telemetry: TelemetryProbe,
    pub(crate) authorizer: Arc<dyn PeerAuthorizer>,
    /// Cancelled when the server stops, which ends its live sessions.
    pub(crate) shutdown: CancellationToken,
}

impl<B: BackendIr> SessionSetup<B> {
    /// The session manager hosting the devices, moving tensors between servers with `transfer`.
    pub(crate) fn manager<T: TensorTransfer<B>>(&self, transfer: Arc<T>) -> SessionManager<B, T> {
        SessionManager::new(self.devices.clone(), transfer)
            .with_custom_ops(self.custom_ops.clone())
            .with_telemetry(self.telemetry.clone())
    }
}
