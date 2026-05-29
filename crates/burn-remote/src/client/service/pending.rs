//! Request/response correlation.

use crate::shared::{RequestId, TaskResponseContent};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::sync::oneshot;

type CallbackMap = Arc<Mutex<HashMap<RequestId, oneshot::Sender<TaskResponseContent>>>>;

/// Correlates response-producing requests with the caller awaiting each one.
///
/// Each response-producing task ([`ReadTensor`](crate::shared::ComputeTask::ReadTensor),
/// `SyncBackend`, `DTypeUsage`) carries a [`RequestId`]; the server echoes it on the
/// response. The runner thread [`register`](Self::register)s a [`oneshot`] callback before
/// sending the task, and the response-demux task delivers the reply through a [`Responder`].
///
/// The callback map is guarded by a plain [`std::sync::Mutex`]: the lock is only ever held
/// for a single insert/remove and never across an `.await`, so neither the runner thread nor
/// the demux task needs the tokio runtime to touch it.
pub(crate) struct PendingResponses {
    callbacks: CallbackMap,
    next_id: RequestId,
}

impl PendingResponses {
    pub(crate) fn new() -> Self {
        Self {
            callbacks: Arc::new(Mutex::new(HashMap::new())),
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
    pub(crate) fn register(&self, id: RequestId) -> oneshot::Receiver<TaskResponseContent> {
        let (tx, rx) = oneshot::channel();
        self.callbacks.lock().unwrap().insert(id, tx);
        rx
    }

    /// A cheap, cloneable handle the response-demux task uses to deliver replies.
    pub(crate) fn responder(&self) -> Responder {
        Responder {
            callbacks: self.callbacks.clone(),
        }
    }
}

/// Delivers responses to the callbacks registered in [`PendingResponses`]. Held by the
/// response-demux task, decoupled from the [`PendingResponses`] the runner thread owns.
pub(crate) struct Responder {
    callbacks: CallbackMap,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn content() -> TaskResponseContent {
        TaskResponseContent::SyncBackend(Ok(()))
    }

    #[test]
    fn next_id_is_monotonic() {
        let mut pending = PendingResponses::new();
        assert_eq!(pending.next_id(), 0);
        assert_eq!(pending.next_id(), 1);
        assert_eq!(pending.next_id(), 2);
    }

    #[test]
    fn register_then_complete_delivers_to_receiver() {
        let mut pending = PendingResponses::new();
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
        let pending = PendingResponses::new();
        assert!(!pending.responder().complete(42, content()));
    }

    #[test]
    fn complete_consumes_the_callback() {
        let mut pending = PendingResponses::new();
        let id = pending.next_id();
        let _rx = pending.register(id);

        assert!(pending.responder().complete(id, content()));
        // The callback was removed on first delivery; a duplicate response finds nothing.
        assert!(!pending.responder().complete(id, content()));
    }
}
