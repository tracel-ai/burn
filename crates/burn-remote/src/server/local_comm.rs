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

/// Rendezvous registry for same-host communication, keyed by [`LocalTransferId`].
pub(crate) struct LocalCommService<B: BackendIr> {
    /// Primitives exposed by source sessions, waiting to be taken by their target.
    pending: Mutex<HashMap<LocalTransferId, HandleKind<B>>>,
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
    /// When `transfer_id` names another session than `source`, the one exposing it, or is already
    /// exposed: either would hand a target the wrong tensor.
    pub async fn expose(
        &self,
        source: SessionId,
        transfer_id: LocalTransferId,
        tensor: HandleKind<B>,
    ) -> Result<(), String> {
        if transfer_id.source() != source {
            return Err(format!(
                "session {source} exposed {transfer_id:?}, which names another session"
            ));
        }
        match self.pending.lock().await.entry(transfer_id) {
            Entry::Occupied(_) => return Err(format!("{transfer_id:?} is exposed already")),
            Entry::Vacant(entry) => {
                entry.insert(tensor);
            }
        }
        self.notify.notify_waiters();
        Ok(())
    }

    /// Take the primitive exposed under `transfer_id`, waiting until it is exposed.
    pub async fn take(&self, transfer_id: LocalTransferId) -> HandleKind<B> {
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

    fn tensor(value: f32) -> HandleKind<Flex> {
        HandleKind::Float(Flex::float_from_data(
            TensorData::from([value]),
            &Default::default(),
        ))
    }

    async fn value(kind: HandleKind<Flex>) -> f32 {
        let HandleKind::Float(tensor) = kind else {
            panic!("a float tensor was exposed");
        };
        Flex::float_into_data(tensor)
            .await
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap()[0]
    }

    #[tokio::test]
    async fn two_sessions_counting_alike_keep_their_own_tensors() {
        let comm = LocalCommService::<Flex>::new();
        let (first, second) = (SessionId::new(), SessionId::new());
        let ids = [first, second].map(|session| LocalTransferId::new(session, 0));

        comm.expose(first, ids[0], tensor(1.0)).await.unwrap();
        comm.expose(second, ids[1], tensor(2.0)).await.unwrap();

        assert_eq!(value(comm.take(ids[1]).await).await, 2.0);
        assert_eq!(value(comm.take(ids[0]).await).await, 1.0);
    }

    #[tokio::test]
    async fn an_id_naming_another_session_is_refused() {
        let comm = LocalCommService::<Flex>::new();
        let other = LocalTransferId::new(SessionId::new(), 0);

        let exposed = comm.expose(SessionId::new(), other, tensor(1.0)).await;

        assert!(exposed.is_err());
        assert!(comm.is_empty().await);
    }

    #[tokio::test]
    async fn a_transfer_exposed_twice_is_refused() {
        let comm = LocalCommService::<Flex>::new();
        let session = SessionId::new();
        let id = LocalTransferId::new(session, 0);

        comm.expose(session, id, tensor(1.0)).await.unwrap();
        let again = comm.expose(session, id, tensor(2.0)).await;

        assert!(again.is_err());
        assert_eq!(value(comm.take(id).await).await, 1.0);
    }
}
