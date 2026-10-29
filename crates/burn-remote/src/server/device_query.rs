use std::{
    collections::HashMap,
    panic::{self, AssertUnwindSafe},
    sync::Arc,
};

use burn_backend::tensor::Device;
use burn_ir::BackendIr;

use crate::shared::{DeviceQueryFailure, DeviceQueryId};

/// Answers a device query (see [`BackendServer::with_device_query`](super::BackendServer::with_device_query)).
type DeviceQueryHandler<B> = Arc<dyn Fn(&Device<B>) -> Vec<u8> + Send + Sync>;

/// The device queries a server answers, by id, shared read-only with every session.
///
/// A query is a side request, not a tensor operation: it asks about the session's device, such as
/// its hardware properties, which the client cannot learn from the operations it sends.
pub(crate) struct DeviceQueries<B: BackendIr> {
    handlers: HashMap<DeviceQueryId, DeviceQueryHandler<B>>,
}

impl<B: BackendIr> DeviceQueries<B> {
    /// Answer `id` with `handler`, replacing the handler `id` had.
    pub(crate) fn register<F>(&mut self, id: DeviceQueryId, handler: F)
    where
        F: Fn(&Device<B>) -> Vec<u8> + Send + Sync + 'static,
    {
        self.handlers.insert(id, Arc::new(handler));
    }

    /// The answer to `id` about `device`. A handler that panics is answered as a failure rather
    /// than taking the session down with it.
    pub(crate) fn answer(
        &self,
        id: &DeviceQueryId,
        device: &Device<B>,
    ) -> Result<Vec<u8>, DeviceQueryFailure> {
        let handler = self.handlers.get(id).ok_or(DeviceQueryFailure::Unknown)?;
        panic::catch_unwind(AssertUnwindSafe(|| handler(device))).map_err(|payload| {
            let reason = payload
                .downcast_ref::<&str>()
                .map(|reason| reason.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "no message".into());
            DeviceQueryFailure::Panicked { reason }
        })
    }
}

impl<B: BackendIr> Default for DeviceQueries<B> {
    fn default() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }
}

impl<B: BackendIr> Clone for DeviceQueries<B> {
    fn clone(&self) -> Self {
        Self {
            handlers: self.handlers.clone(),
        }
    }
}
