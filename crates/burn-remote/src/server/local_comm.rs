//! In-process communication between two sessions hosted by the same server.
//!
//! Whenever sessions on the **same** server (same address, different device index) need to
//! hand device-resident data to each other, there is no reason to round-trip through the host:
//! both sessions' [`TensorInterpreter`](burn_router::TensorInterpreter)s live in the same
//! process, so the source can hand its primitive straight to the target, which moves it onto
//! its own device with the inner backend's `to_device`. This is the same-host counterpart of the
//! cross-server tensor transfer, which moves data *between* servers over the network.
//!
//! Today this backs cross-device `to_device`, but the mechanism is general: any same-host
//! collective (e.g. all-reduce) that needs ranks to exchange primitives uses the same
//! expose/take rendezvous.
//!
//! This registry is the rendezvous point. The source session exposes a primitive under a
//! [`LocalTransferId`]; the target session takes it. Either may arrive first — the source op
//! and the target op travel on separate connections with no cross-connection ordering — so the
//! taker waits on a [`Notify`] until the primitive shows up.

use crate::shared::{LocalTransferId, SessionId};
use burn_ir::{BackendIr, HandleKind};
use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

/// What a source session left under a transfer id.
enum Exposed<B: BackendIr> {
    Tensor(HandleKind<B>),
    /// Exposed more than once, so no target can tell which tensor is its own.
    Ambiguous,
}

/// Rendezvous registry for same-host communication, keyed by [`LocalTransferId`].
pub(crate) struct LocalCommService<B: BackendIr> {
    /// Primitives exposed by source sessions, waiting to be taken by their target.
    pending: Mutex<HashMap<LocalTransferId, Exposed<B>>>,
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

    /// Expose `tensor` under `transfer_id`, waking any session already waiting for it.
    ///
    /// # Errors
    ///
    /// When `transfer_id` is exposed already. Its takers then fail as well, since none can tell
    /// which tensor is its own.
    pub async fn expose(
        &self,
        transfer_id: LocalTransferId,
        tensor: HandleKind<B>,
    ) -> Result<(), String> {
        let exposed = match self.pending.lock().await.entry(transfer_id) {
            Entry::Occupied(mut entry) => {
                entry.insert(Exposed::Ambiguous);
                Err(format!("{transfer_id:?} is exposed already"))
            }
            Entry::Vacant(entry) => {
                entry.insert(Exposed::Tensor(tensor));
                Ok(())
            }
        };
        self.notify.notify_waiters();
        exposed
    }

    /// Take the primitive exposed under `transfer_id`, waiting until it is exposed.
    ///
    /// # Errors
    ///
    /// When `transfer_id` was exposed more than once.
    pub async fn take(&self, transfer_id: LocalTransferId) -> Result<HandleKind<B>, String> {
        loop {
            // Register interest *before* checking the map so an `expose` that lands between
            // the check and the await can't slip through the gap and leave us parked forever.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let mut pending = self.pending.lock().await;
            match pending.remove(&transfer_id) {
                Some(Exposed::Tensor(tensor)) => return Ok(tensor),
                Some(Exposed::Ambiguous) => {
                    pending.insert(transfer_id, Exposed::Ambiguous);
                    return Err(format!("{transfer_id:?} was exposed more than once"));
                }
                None => drop(pending),
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
            .retain(|transfer_id, _| transfer_id.source() != session);
    }

    #[cfg(all(test, not(target_family = "wasm")))]
    pub(crate) async fn is_empty(&self) -> bool {
        self.pending.lock().await.is_empty()
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use burn_backend::{TensorData, ops::FloatTensorOps};
    use burn_flex::Flex;
    use std::time::Duration;

    fn tensor(value: f32) -> HandleKind<Flex> {
        HandleKind::Float(Flex::float_from_data(
            TensorData::from([value]),
            &Default::default(),
        ))
    }

    #[tokio::test]
    async fn a_transfer_exposed_twice_fails_each_taker() {
        let comm = LocalCommService::<Flex>::new();
        let id = LocalTransferId::new(SessionId::new(), 0);

        comm.expose(id, tensor(1.0)).await.unwrap();
        let again = comm.expose(id, tensor(2.0)).await;

        assert!(again.is_err());
        for _ in 0..2 {
            let taken = tokio::time::timeout(Duration::from_secs(10), comm.take(id)).await;
            assert!(
                matches!(taken, Ok(Err(_))),
                "a taker of a transfer exposed twice did not fail"
            );
        }
    }
}
