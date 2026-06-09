//! In-process communication between two sessions hosted by the same server.
//!
//! Whenever sessions on the **same** server (same address, different device index) need to
//! hand device-resident data to each other, there is no reason to round-trip through the host:
//! both sessions' [`TensorInterpreter`](burn_router::TensorInterpreter)s live in the same
//! process, so the source can hand its primitive straight to the target, which moves it onto
//! its own device with the inner backend's `to_device`. This is the same-host counterpart of
//! the [`external_comm`](burn_communication::external_comm) service, which moves data *between*
//! servers over the network.
//!
//! Today this backs cross-device `to_device`, but the mechanism is general: any same-host
//! collective (e.g. all-reduce) that needs ranks to exchange primitives uses the same
//! expose/take rendezvous.
//!
//! This registry is the rendezvous point. The source session exposes a primitive under a
//! [`TensorTransferId`]; the target session takes it. Either may arrive first — the source op
//! and the target op travel on separate connections with no cross-connection ordering — so the
//! taker waits on a [`Notify`] until the primitive shows up.

use crate::shared::SessionId;
use burn_communication::external_comm::TensorTransferId;
use burn_ir::{BackendIr, HandleKind};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

/// A primitive parked in the registry, tagged with the session that exposed it so the entry can
/// be reclaimed if that session goes away before its target takes it.
struct Pending<B: BackendIr> {
    source: SessionId,
    tensor: HandleKind<B>,
}

/// Rendezvous registry for same-host communication, keyed by [`TensorTransferId`].
pub(crate) struct LocalCommService<B: BackendIr> {
    /// Primitives exposed by source sessions, waiting to be taken by their target.
    pending: Mutex<HashMap<TensorTransferId, Pending<B>>>,
    /// Wakes takers whenever a new primitive is exposed.
    notify: Arc<Notify>,
}

impl<B: BackendIr> LocalCommService<B> {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Expose `tensor` under `transfer_id`, waking any session already waiting for it. `source` is
    /// the session exposing it, so a stranded entry can be purged if that session closes before its
    /// target takes the tensor (see [`purge_session`](Self::purge_session)).
    pub async fn expose(
        &self,
        source: SessionId,
        transfer_id: TensorTransferId,
        tensor: HandleKind<B>,
    ) {
        self.pending
            .lock()
            .await
            .insert(transfer_id, Pending { source, tensor });
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

            if let Some(pending) = self.pending.lock().await.remove(&transfer_id) {
                return pending.tensor;
            }

            notified.await;
        }
    }

    /// Drop every primitive `session` exposed that no target ever took.
    ///
    /// Called when a session closes. In normal operation every exposed primitive is taken (and so
    /// removed) by its target, but if the source session tears down first — a crash, a dropped
    /// connection, a transfer whose target never runs — its entry would otherwise sit in the map
    /// for the server's lifetime, pinning device memory. Purging by source session reclaims it.
    pub async fn purge_session(&self, session: SessionId) {
        self.pending
            .lock()
            .await
            .retain(|_, pending| pending.source != session);
    }
}
