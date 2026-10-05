//! Request/response correlation.

use super::registry::SessionEnd;
use crate::shared::{RequestId, TaskResponseContent};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::sync::oneshot;

/// The callbacks awaiting replies. Registering checks the session's end under this lock, so none
/// slips in after [`Responder::end_session`] has drained them.
type SharedCallbacks = Arc<Mutex<HashMap<RequestId, oneshot::Sender<TaskResponseContent>>>>;

/// Correlates response-producing requests with the caller awaiting each one.
///
/// Each response-producing task ([`ReadTensor`](crate::shared::Task::ReadTensor),
/// `SyncBackend`, `DTypeUsage`, `ProfileStart`, `ProfileEnd`) carries a [`RequestId`]; the
/// server echoes it on the
/// response. The runner thread [`register`](Self::register)s a [`oneshot`] callback before
/// sending the task, and the response-demux task delivers the reply through a [`Responder`].
///
/// The callbacks are guarded by a plain [`std::sync::Mutex`]: the lock is only ever held for a
/// single insert/remove/drain and never across an `.await`, so neither the runner thread nor the
/// demux task needs the tokio runtime to touch it.
pub(crate) struct PendingResponses {
    callbacks: SharedCallbacks,
    session: Arc<SessionEnd>,
    next_id: RequestId,
}

impl PendingResponses {
    pub(crate) fn new(session: Arc<SessionEnd>) -> Self {
        Self {
            callbacks: SharedCallbacks::default(),
            session,
            next_id: 0,
        }
    }

    /// Allocate the next monotonic [`RequestId`].
    pub(crate) fn next_id(&mut self) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Register a callback for `id`, returning the receiver the caller awaits for the reply.
    ///
    /// Once the session has ended, the receiver resolves at once to a `RecvError`.
    pub(crate) fn register(&self, id: RequestId) -> oneshot::Receiver<TaskResponseContent> {
        let (tx, rx) = oneshot::channel();
        let mut callbacks = self.callbacks.lock().unwrap();
        if !self.session.has_ended() {
            callbacks.insert(id, tx);
        }
        rx
    }

    /// A cheap, cloneable handle the response-demux task uses to deliver replies.
    pub(crate) fn responder(&self) -> Responder {
        Responder {
            callbacks: self.callbacks.clone(),
            session: self.session.clone(),
        }
    }
}

/// Delivers responses to the callbacks registered in [`PendingResponses`]. Held by the
/// response-demux task, decoupled from the [`PendingResponses`] the runner thread owns.
#[derive(Clone)]
pub(crate) struct Responder {
    callbacks: SharedCallbacks,
    session: Arc<SessionEnd>,
}

impl Responder {
    /// Deliver `content` to the caller waiting on `id`. Returns `false` if no callback is
    /// registered (unknown id, or the caller dropped its receiver), in which case the
    /// response is discarded.
    pub(crate) fn complete(&self, id: RequestId, content: TaskResponseContent) -> bool {
        match self.callbacks.lock().unwrap().remove(&id) {
            Some(tx) => {
                // Receiver dropped is fine (caller no longer cares).
                let _ = tx.send(content);
                true
            }
            None => false,
        }
    }

    /// End the session, and fail every caller waiting on a reply or asking for one later.
    pub(crate) fn end_session(&self) {
        // Before any caller wakes, so one that connects again on the failure gets a new device.
        self.session.end();
        self.callbacks.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending_responses() -> PendingResponses {
        PendingResponses::new(Arc::default())
    }

    fn content() -> TaskResponseContent {
        TaskResponseContent::SyncBackend(Ok(()))
    }

    #[test]
    fn next_id_is_monotonic() {
        let mut pending = pending_responses();
        assert_eq!(pending.next_id(), 0);
        assert_eq!(pending.next_id(), 1);
        assert_eq!(pending.next_id(), 2);
    }

    #[test]
    fn register_then_complete_delivers_to_receiver() {
        let mut pending = pending_responses();
        let id = pending.next_id();
        let mut rx = pending.register(id);

        assert!(pending.responder().complete(id, content()));
        // `try_recv` resolves synchronously once the sender has fired — no runtime needed.
        assert!(matches!(
            rx.try_recv(),
            Ok(TaskResponseContent::SyncBackend(Ok(())))
        ));
    }

    #[test]
    fn complete_unknown_id_returns_false() {
        let pending = pending_responses();
        assert!(!pending.responder().complete(42, content()));
    }

    #[test]
    fn complete_consumes_the_callback() {
        let mut pending = pending_responses();
        let id = pending.next_id();
        let _rx = pending.register(id);

        assert!(pending.responder().complete(id, content()));
        // The callback was removed on first delivery; a duplicate response finds nothing.
        assert!(!pending.responder().complete(id, content()));
    }

    #[test]
    fn end_session_fails_every_pending_caller() {
        let session = Arc::new(SessionEnd::default());
        let mut pending = PendingResponses::new(session.clone());
        let id0 = pending.next_id();
        let id1 = pending.next_id();
        let mut rx0 = pending.register(id0);
        let mut rx1 = pending.register(id1);

        // The response stream died with both requests still in flight.
        pending.responder().end_session();
        assert!(session.has_ended());

        // Both receivers resolve immediately with an error instead of hanging.
        assert!(matches!(
            rx0.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert!(matches!(
            rx1.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
    }

    #[test]
    fn register_after_the_session_ended_returns_a_closed_receiver() {
        let mut pending = pending_responses();
        pending.responder().end_session();

        // A request issued after the connection dropped must not park forever.
        let id = pending.next_id();
        let mut rx = pending.register(id);
        assert!(matches!(
            rx.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));

        // And it was never inserted, so a late response for it finds nothing.
        assert!(!pending.responder().complete(id, content()));
    }
}
