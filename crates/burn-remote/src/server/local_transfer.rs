//! In-process tensor transfers between two sessions hosted by the same server.
//!
//! When a client moves a tensor between two devices that live on the **same** server (same
//! address, different device index), there is no reason to round-trip the data through the
//! host: both sessions' [`TensorInterpreter`](burn_router::TensorInterpreter)s are in the same
//! process, so the source can hand its device-resident primitive straight to the target, which
//! moves it onto its own device with the inner backend's `to_device`.
//!
//! This registry is the rendezvous point. The source session exposes a primitive under a
//! [`TensorTransferId`]; the target session takes it. Either may arrive first — the source op
//! and the target op travel on separate connections with no cross-connection ordering — so the
//! taker waits on a [`Notify`] until the primitive shows up.

use burn_communication::data_service::TensorTransferId;
use burn_ir::{BackendIr, HandleKind};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

/// Rendezvous registry for same-host transfers, keyed by [`TensorTransferId`].
pub(crate) struct LocalTransferService<B: BackendIr> {
    /// Primitives exposed by source sessions, waiting to be taken by their target.
    pending: Mutex<HashMap<TensorTransferId, HandleKind<B>>>,
    /// Wakes takers whenever a new primitive is exposed.
    notify: Arc<Notify>,
}

impl<B: BackendIr> LocalTransferService<B> {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Expose `tensor` under `transfer_id`, waking any session already waiting for it.
    pub async fn expose(&self, transfer_id: TensorTransferId, tensor: HandleKind<B>) {
        self.pending.lock().await.insert(transfer_id, tensor);
        self.notify.notify_waiters();
    }

    /// Take the primitive exposed under `transfer_id`, waiting until it is exposed.
    pub async fn take(&self, transfer_id: TensorTransferId) -> HandleKind<B> {
        loop {
            // Register interest *before* checking the map so an `expose` that lands between
            // the check and the await can't slip through the gap and leave us parked forever.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            if let Some(tensor) = self.pending.lock().await.remove(&transfer_id) {
                return tensor;
            }

            notified.await;
        }
    }
}
