//! The socket-free half of the submit handler: deciding what to do with each decoded message
//! and driving the session layer, with no dependency on the wire. Kept separate from the socket
//! so it can be driven directly in tests.

use std::sync::Arc;

use tokio::sync::mpsc;

use super::submit::SubmitService;
use crate::shared::{RemoteMessage, SessionId, Task};

pub(super) struct SubmitPolicy<S> {
    service: Arc<S>,
    state: SubmitState,
}

impl<S: SubmitService> SubmitPolicy<S> {
    pub(super) fn new(service: Arc<S>) -> Self {
        Self {
            service,
            state: SubmitState::default(),
        }
    }

    /// Process one decoded batch in order. Each websocket frame may carry a batch of messages
    /// (the client buffers fire-and-forget tasks before sending). Returns `true` if the
    /// connection should close — a `Close` short-circuits the rest of the batch, matching the
    /// client side (once it sends `Close`, it sends nothing after).
    pub(super) async fn process_batch(&mut self, messages: Vec<RemoteMessage>) -> bool {
        for message in messages {
            match self.state.decide(message) {
                Action::Continue => {}
                Action::Forward { session_id, task } => {
                    if !self.forward(session_id, task).await {
                        return true;
                    }
                }
                Action::Close(id) => {
                    log::debug!("Close requested for session {id}");
                    self.service.close(id).await;
                    return true;
                }
                Action::Abort => {
                    log::error!("Task submitted before session Init; closing submit stream");
                    return true;
                }
            }
        }
        false
    }

    /// Forward a task to the session worker, resolving and caching the worker channel on first
    /// use. Returns `false` if the worker is gone (the connection should close), which shouldn't
    /// happen while the session is live.
    ///
    /// A full channel applies async backpressure here — the `send` await yields and we stop
    /// reading the socket — rather than blocking a runtime thread.
    async fn forward(&mut self, session_id: SessionId, task: Task) -> bool {
        if self.state.task_sender.is_none() {
            self.state.task_sender = Some(
                self.service
                    .session_task_sender(session_id, self.state.device_index)
                    .await,
            );
        }
        let sender = self.state.task_sender.as_ref().unwrap();

        if sender.send(task).await.is_err() {
            log::error!("Session {session_id} worker channel closed; closing submit stream");
            return false;
        }
        true
    }

    /// Tear the session down once the submit loop ends. The worker, runner, and result queue are
    /// released because closing the session — and dropping our cached task sender on return —
    /// closes the worker's channel, so it flushes and exits, which in turn closes the fetch
    /// writer's queue.
    pub(super) async fn cleanup(&self) {
        match self.state.session_id {
            Some(id) if !self.state.session_closed => {
                // The stream ended without an explicit `Close` (client crash, dropped socket,
                // decode/stream error). Close here so nothing leaks.
                log::debug!("Submit stream for session {id} ended without Close; cleaning up");
                self.service.close(id).await;
            }
            Some(id) => log::debug!("Closing session {id}"),
            None => log::debug!("Closing session (no id info)"),
        }
    }
}

/// Which session a submit connection is bound to, plus the channel to that session's worker.
///
/// A single connection serves one session at a time but may be re-bound to a new session by a
/// later `RemoteMessage::Init` (the client never does this today, but the protocol allows it),
/// which is why the session id is optional and the worker channel is re-resolved on rebind.
#[derive(Default)]
struct SubmitState {
    /// The session this connection is currently bound to, set by `RemoteMessage::Init`.
    session_id: Option<SessionId>,
    /// The device index the bound session is pinned to, fixed by `RemoteMessage::Init`.
    device_index: u32,
    /// Whether an explicit `RemoteMessage::Close` already tore the session down, so the final
    /// cleanup doesn't double-close.
    session_closed: bool,
    /// The bound session's worker channel, resolved on the first task and reused for the rest of
    /// the connection. Cleared on rebind so the new session is resolved afresh.
    task_sender: Option<mpsc::Sender<Task>>,
}

/// What the submit loop should do with a single decoded [`RemoteMessage`].
///
/// `Forward` carries a whole `Task`, so it dwarfs the other variants — but an `Action` is
/// returned by `decide` and matched immediately, never stored or collected, so the size
/// disparity is free. Boxing the task instead would add a heap allocation on every op.
#[allow(clippy::large_enum_variant)]
enum Action {
    /// State-only message (an `Init` rebind); move on to the next message in the batch.
    Continue,
    /// Forward this task to the bound session's worker.
    Forward { session_id: SessionId, task: Task },
    /// Explicit `Close` for `session_id`: tear the session down and end the connection.
    Close(SessionId),
    /// Protocol violation (a task submitted before any `Init`): end the connection.
    Abort,
}

impl SubmitState {
    /// Advance the connection state by one message and report what the loop should do with it.
    /// Pure: it touches no socket, service, or backend, so the whole submit policy can be
    /// exercised in isolation.
    fn decide(&mut self, message: RemoteMessage) -> Action {
        match message {
            RemoteMessage::Init(id, index) => {
                self.session_id = Some(id);
                self.device_index = index;
                // Re-resolve the worker channel for the (re)bound session on the next task.
                self.task_sender = None;
                Action::Continue
            }
            RemoteMessage::Close(id) => {
                self.session_closed = true;
                Action::Close(id)
            }
            RemoteMessage::Task(task) => match self.session_id {
                Some(session_id) => Action::Forward { session_id, task },
                None => Action::Abort,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::Mutex;

    fn block_on<F: Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(fut)
    }

    /// A fake [`SubmitService`] that records what it was asked to do: it hands out a single
    /// worker channel (whose receiver the test holds) and logs every `close`.
    struct FakeService {
        task_tx: mpsc::Sender<Task>,
        closed: Mutex<Vec<SessionId>>,
    }

    impl FakeService {
        fn new() -> (Arc<Self>, mpsc::Receiver<Task>) {
            let (task_tx, task_rx) = mpsc::channel(16);
            let service = Arc::new(Self {
                task_tx,
                closed: Mutex::new(Vec::new()),
            });
            (service, task_rx)
        }

        fn closed(&self) -> Vec<SessionId> {
            self.closed.lock().unwrap().clone()
        }
    }

    impl SubmitService for FakeService {
        async fn session_task_sender(
            &self,
            _session_id: SessionId,
            _device_index: u32,
        ) -> mpsc::Sender<Task> {
            self.task_tx.clone()
        }

        async fn close(&self, session_id: SessionId) {
            self.closed.lock().unwrap().push(session_id);
        }
    }

    fn drain<T>(rx: &mut mpsc::Receiver<T>) -> Vec<T> {
        let mut out = Vec::new();
        while let Ok(item) = rx.try_recv() {
            out.push(item);
        }
        out
    }

    // --- SubmitState::decide (pure) -------------------------------------------------------

    #[test]
    fn decide_init_binds_session_and_resets_worker_channel() {
        let mut state = SubmitState::default();
        // A worker channel left over from a previous binding must be dropped on rebind.
        state.task_sender = Some(mpsc::channel(1).0);

        let id = SessionId::new();
        let action = state.decide(RemoteMessage::Init(id, 5));

        assert!(matches!(action, Action::Continue));
        assert_eq!(state.session_id, Some(id));
        assert_eq!(state.device_index, 5);
        assert!(state.task_sender.is_none());
        assert!(!state.session_closed);
    }

    #[test]
    fn decide_task_before_init_aborts() {
        let mut state = SubmitState::default();
        let action = state.decide(RemoteMessage::Task(Task::Seed(1)));
        assert!(matches!(action, Action::Abort));
    }

    #[test]
    fn decide_task_after_init_forwards_to_bound_session() {
        let mut state = SubmitState::default();
        let id = SessionId::new();
        state.decide(RemoteMessage::Init(id, 0));

        match state.decide(RemoteMessage::Task(Task::Seed(7))) {
            Action::Forward { session_id, task } => {
                assert_eq!(session_id, id);
                assert!(matches!(task, Task::Seed(7)));
            }
            _ => panic!("expected a forward to the bound session"),
        }
    }

    #[test]
    fn decide_close_marks_the_session_closed() {
        let mut state = SubmitState::default();
        let id = SessionId::new();
        state.decide(RemoteMessage::Init(id, 0));

        let action = state.decide(RemoteMessage::Close(id));

        assert!(matches!(action, Action::Close(closed) if closed == id));
        assert!(state.session_closed);
    }

    // --- SubmitPolicy (against the fake service) ------------------------------------------

    #[test]
    fn policy_forwards_pre_close_tasks_and_stops_at_close() {
        block_on(async {
            let (service, mut task_rx) = FakeService::new();
            let mut policy = SubmitPolicy::new(service.clone());
            let id = SessionId::new();

            let stop = policy
                .process_batch(vec![
                    RemoteMessage::Init(id, 0),
                    RemoteMessage::Task(Task::Seed(1)),
                    RemoteMessage::Task(Task::Seed(2)),
                    RemoteMessage::Close(id),
                    // After `Close` the batch short-circuits: this must never be forwarded.
                    RemoteMessage::Task(Task::Seed(3)),
                ])
                .await;

            assert!(stop, "an explicit Close ends the connection");
            assert_eq!(service.closed(), vec![id]);

            let forwarded = drain(&mut task_rx);
            assert_eq!(forwarded.len(), 2, "only the two pre-Close tasks forward");
            assert!(matches!(forwarded[0], Task::Seed(1)));
            assert!(matches!(forwarded[1], Task::Seed(2)));
        });
    }

    #[test]
    fn policy_aborts_on_task_before_init() {
        block_on(async {
            let (service, _rx) = FakeService::new();
            let mut policy = SubmitPolicy::new(service.clone());

            let stop = policy
                .process_batch(vec![RemoteMessage::Task(Task::Seed(1))])
                .await;

            assert!(stop, "a task before any Init ends the connection");
            assert!(service.closed().is_empty(), "nothing to close yet");
        });
    }

    #[test]
    fn policy_cleanup_closes_a_session_that_ended_without_close() {
        block_on(async {
            let (service, _rx) = FakeService::new();
            let mut policy = SubmitPolicy::new(service.clone());
            let id = SessionId::new();

            // The batch never sends `Close` — simulate an abrupt disconnect afterwards.
            policy
                .process_batch(vec![
                    RemoteMessage::Init(id, 0),
                    RemoteMessage::Task(Task::Seed(1)),
                ])
                .await;
            policy.cleanup().await;

            assert_eq!(service.closed(), vec![id]);
        });
    }

    #[test]
    fn policy_cleanup_does_not_double_close_after_explicit_close() {
        block_on(async {
            let (service, _rx) = FakeService::new();
            let mut policy = SubmitPolicy::new(service.clone());
            let id = SessionId::new();

            policy
                .process_batch(vec![RemoteMessage::Init(id, 0), RemoteMessage::Close(id)])
                .await;
            policy.cleanup().await;

            assert_eq!(service.closed(), vec![id], "closed exactly once");
        });
    }
}
