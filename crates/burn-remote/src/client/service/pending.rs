//! Request/response correlation.

use crate::shared::{RequestId, TaskResponseContent};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::sync::oneshot;

/// The callbacks awaiting replies, plus whether the connection is still live.
///
/// Both live under one mutex so that registering a callback and tearing every callback down on
/// disconnect are mutually exclusive: a registration can't slip into the map *after* the
/// response-demux task has already drained it on disconnect and left it to wait for a reply that
/// will never come.
struct State {
    callbacks: HashMap<RequestId, oneshot::Sender<TaskResponseContent>>,
    /// Cleared by [`Responder::disconnect`] once the response stream is gone; gates new
    /// registrations so a post-disconnect request fails fast instead of parking forever.
    connected: bool,
}

type SharedState = Arc<Mutex<State>>;

/// Correlates response-producing requests with the caller awaiting each one.
///
/// Each response-producing task ([`ReadTensor`](crate::shared::Task::ReadTensor),
/// `SyncBackend`, `DTypeUsage`) carries a [`RequestId`]; the server echoes it on the
/// response. The runner thread [`register`](Self::register)s a [`oneshot`] callback before
/// sending the task, and the response-demux task delivers the reply through a [`Responder`].
///
/// The state is guarded by a plain [`std::sync::Mutex`]: the lock is only ever held for a single
/// insert/remove/drain and never across an `.await`, so neither the runner thread nor the demux
/// task needs the tokio runtime to touch it.
pub(crate) struct PendingResponses {
    state: SharedState,
    next_id: RequestId,
}

impl PendingResponses {
    pub(crate) fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                callbacks: HashMap::new(),
                connected: true,
            })),
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
    /// If the connection has already dropped (the demux task called [`Responder::disconnect`]),
    /// the sender is dropped immediately, so the returned receiver resolves to a `RecvError` right
    /// away rather than blocking on a response the dead server will never send.
    pub(crate) fn register(&self, id: RequestId) -> oneshot::Receiver<TaskResponseContent> {
        let (tx, rx) = oneshot::channel();
        let mut state = self.state.lock().unwrap();
        if state.connected {
            state.callbacks.insert(id, tx);
        }
        // Disconnected: drop `tx` here, leaving `rx` already-closed.
        rx
    }

    /// A cheap, cloneable handle the response-demux task uses to deliver replies.
    pub(crate) fn responder(&self) -> Responder {
        Responder {
            state: self.state.clone(),
        }
    }
}

/// Delivers responses to the callbacks registered in [`PendingResponses`]. Held by the
/// response-demux task, decoupled from the [`PendingResponses`] the runner thread owns.
pub(crate) struct Responder {
    state: SharedState,
}

impl Responder {
    /// Deliver `content` to the caller waiting on `id`. Returns `false` if no callback is
    /// registered (unknown id, or the caller dropped its receiver), in which case the
    /// response is discarded.
    pub(crate) fn complete(&self, id: RequestId, content: TaskResponseContent) -> bool {
        match self.state.lock().unwrap().callbacks.remove(&id) {
            Some(tx) => {
                // Receiver dropped is fine (caller no longer cares).
                let _ = tx.send(content);
                true
            }
            None => false,
        }
    }

    /// Mark the connection dropped and fail every pending caller.
    ///
    /// Called by the response-demux task when the response stream closes or errors (server down,
    /// connection reset). Clearing the map drops every registered sender, so each waiting receiver
    /// resolves to a `RecvError` that the callers (`sync`, `read_tensor`, `dtype_usage`) translate
    /// into an error instead of blocking forever. Clearing `connected` makes any later request
    /// fail fast in [`PendingResponses::register`] rather than registering a doomed callback.
    pub(crate) fn disconnect(&self) {
        let mut state = self.state.lock().unwrap();
        state.connected = false;
        state.callbacks.clear();
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

    #[test]
    fn disconnect_fails_every_pending_caller() {
        let mut pending = PendingResponses::new();
        let id0 = pending.next_id();
        let id1 = pending.next_id();
        let mut rx0 = pending.register(id0);
        let mut rx1 = pending.register(id1);

        // The response stream died with both requests still in flight.
        pending.responder().disconnect();

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
    fn register_after_disconnect_returns_a_closed_receiver() {
        let mut pending = PendingResponses::new();
        pending.responder().disconnect();

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
