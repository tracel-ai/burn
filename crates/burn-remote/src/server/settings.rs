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
    AllowAll, PeerAuthorizer, ServeError, device_query::DeviceQueries, session::SessionManager,
    transfer::TensorTransfer,
};
use crate::{
    shared::DeviceQueryId,
    telemetry::{CHANNEL_CAPACITY, TelemetryProbe},
};

/// What a server applies to every session, whatever its transport: its authorizer, its telemetry,
/// its custom operations and its device queries.
///
/// Custom operations and device queries are kept per backend and checked against the devices'
/// backend when the server starts, so a server that learns its backend from its devices can hold
/// them.
#[doc(hidden)]
#[derive(Clone)]
pub struct ServerSettings {
    authorizer: Arc<dyn PeerAuthorizer>,
    telemetry: Option<TelemetryProbe>,
    backends: HashMap<TypeId, AnyBackendHandlers>,
}

/// One backend's [`BackendHandlers`], with the backend's type erased and its name kept.
#[derive(Clone)]
struct AnyBackendHandlers {
    backend: &'static str,
    handlers: Arc<dyn Any + Send + Sync>,
}

/// What a server runs on one backend's devices for its clients.
struct BackendHandlers<B: BackendIr> {
    custom_ops: CustomOpRegistry<B>,
    queries: DeviceQueries<B>,
}

impl<B: BackendIr> Default for BackendHandlers<B> {
    fn default() -> Self {
        Self {
            custom_ops: CustomOpRegistry::default(),
            queries: DeviceQueries::default(),
        }
    }
}

impl<B: BackendIr> Clone for BackendHandlers<B> {
    fn clone(&self) -> Self {
        Self {
            custom_ops: self.custom_ops.clone(),
            queries: self.queries.clone(),
        }
    }
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            authorizer: Arc::new(AllowAll),
            telemetry: None,
            backends: HashMap::new(),
        }
    }
}

impl ServerSettings {
    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.authorizer = Arc::new(authorizer);
        self
    }

    pub fn with_telemetry(mut self, probe: TelemetryProbe) -> Self {
        self.telemetry = Some(probe);
        self
    }

    pub fn with_custom_op<B: BackendIr, F>(self, id: &str, handler: F) -> Self
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        let mut handlers = self.handlers_of::<B>();
        handlers.custom_ops.register(id, handler);
        self.with_handlers(handlers)
    }

    pub fn with_custom_ops<B: BackendIr>(self, registry: CustomOpRegistry<B>) -> Self {
        let mut handlers = self.handlers_of::<B>();
        handlers.custom_ops = registry;
        self.with_handlers(handlers)
    }

    pub fn with_device_query<B: BackendIr, F>(self, id: DeviceQueryId, handler: F) -> Self
    where
        F: Fn(&Device<B>) -> Vec<u8> + Send + Sync + 'static,
    {
        let mut handlers = self.handlers_of::<B>();
        handlers.queries.register(id, handler);
        self.with_handlers(handlers)
    }

    fn with_handlers<B: BackendIr>(mut self, handlers: BackendHandlers<B>) -> Self {
        self.backends.insert(
            TypeId::of::<B>(),
            AnyBackendHandlers {
                backend: type_name::<B>(),
                handlers: Arc::new(handlers),
            },
        );
        self
    }

    fn handlers_of<B: BackendIr>(&self) -> BackendHandlers<B> {
        self.backends
            .get(&TypeId::of::<B>())
            .and_then(|other| other.handlers.downcast_ref::<BackendHandlers<B>>())
            .cloned()
            .unwrap_or_default()
    }

    /// Everything a session of `devices` is set up with, checking that the custom operations and
    /// device queries are for their backend.
    pub(crate) fn sessions_of<B: BackendIr>(
        &self,
        devices: Vec<Device<B>>,
        shutdown: CancellationToken,
    ) -> Result<SessionSetup<B>, ServeError> {
        if devices.is_empty() {
            return Err(ServeError::NoDevices);
        }
        if let Some(other) = self
            .backends
            .iter()
            .find_map(|(backend, handlers)| (*backend != TypeId::of::<B>()).then_some(handlers))
        {
            return Err(ServeError::HandlerBackend {
                backend: other.backend,
            });
        }
        // Readbacks on a server's async runtime must not park a blocking device-to-host copy on
        // an executor worker; this makes them eager, for the whole process.
        burn_std::set_runtime_kind(burn_std::RuntimeKind::Async);
        Ok(SessionSetup {
            devices,
            handlers: self.handlers_of::<B>(),
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
    handlers: BackendHandlers<B>,
    telemetry: TelemetryProbe,
    pub(crate) authorizer: Arc<dyn PeerAuthorizer>,
    /// Cancelled when the server stops, which ends its live sessions.
    pub(crate) shutdown: CancellationToken,
}

impl<B: BackendIr> SessionSetup<B> {
    /// The session manager hosting the devices, moving tensors between servers with `transfer`.
    pub(crate) fn manager<T: TensorTransfer<B>>(&self, transfer: Arc<T>) -> SessionManager<B, T> {
        SessionManager::new(self.devices.clone(), transfer)
            .with_custom_ops(self.handlers.custom_ops.clone())
            .with_device_queries(self.handlers.queries.clone())
            .with_telemetry(self.telemetry.clone())
    }
}
