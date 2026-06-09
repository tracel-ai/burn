//! The two websocket connection handlers, one per route.
//!
//! A client opens a pair of sockets per session: a `/request` socket carrying the task
//! stream and a `/response` socket carrying results back. [`SocketRequest`] drives the former
//! and [`SocketResponse`] the latter. Both are otherwise plain loops; the only real state is
//! on the request side, where a connection tracks which session it is bound to and the
//! channel to that session's worker.
//!
//! Both handlers are written against two abstractions rather than concrete types: the socket
//! is a [`CommunicationChannel`], and the session layer they talk to is a [`RequestService`] /
//! [`ResponseService`]. The production service is the [`SessionManager`](super::session::SessionManager),
//! but the traits let the request dispatch be exercised against a fake service with no backend
//! and no live socket (see this module's tests).

use std::future::Future;
use std::sync::Arc;

use burn_communication::{CommunicationChannel, Message};
use burn_std::DeviceSettings;
use tokio::sync::mpsc;

use crate::shared::{ComputeTask, SessionId, Task, TaskResponse, TaskResponseContent};

/// What a `/request` connection needs from the session layer: the worker channel for a
/// session, and a way to tear a session down. Async methods return `impl Future + Send` (as
/// the [`CommunicationChannel`] trait does) so a handler future built on them stays `Send` and
/// can be spawned by the server.
pub(crate) trait RequestService: Send + Sync + 'static {
    /// The channel forwarding compute tasks to `session_id`'s worker, creating the session
    /// (and spawning its worker) on demand.
    fn session_task_sender(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> impl Future<Output = mpsc::Sender<ComputeTask>> + Send;

    /// Drop the session, letting its worker drain and exit. A `close` for an unknown session
    /// is a no-op.
    fn close(&self, session_id: SessionId) -> impl Future<Output = ()> + Send;
}

/// What a `/response` connection needs from the session layer: the device metadata returned on
/// the init handshake, and the session's response receiver to drain.
pub(crate) trait ResponseService: Send + Sync + 'static {
    /// The default settings of the device at `device_index`, returned on the handshake.
    fn device_settings(&self, device_index: u32) -> DeviceSettings;

    /// The number of devices this server hosts, returned on the handshake so the client can
    /// enumerate every device behind the address.
    fn device_count(&self) -> u32;

    /// Claim the session's response receiver. Errors if a responder is already registered —
    /// the protocol allows only one response socket per session.
    fn take_response_receiver(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> impl Future<Output = Result<mpsc::Receiver<TaskResponse>, String>> + Send;
}

/// The `/response` connection: answer the init handshake, then drain the session's response
/// queue onto the socket until it closes.
pub(crate) struct SocketResponse<S, C> {
    service: Arc<S>,
    socket: C,
}

impl<S: ResponseService, C: CommunicationChannel> SocketResponse<S, C> {
    pub(crate) fn new(service: Arc<S>, socket: C) -> Self {
        Self { service, socket }
    }

    pub(crate) async fn run(mut self) {
        log::info!("[Response Handler] On new connection.");

        let Some((session_id, device_index)) = self.handshake().await else {
            return;
        };

        // Claim the session's response receiver. The protocol allows only one response socket
        // per session, so a second responder is rejected here.
        let mut receiver = match self
            .service
            .take_response_receiver(session_id, device_index)
            .await
        {
            Ok(r) => r,
            Err(err) => {
                log::error!("{err}");
                return;
            }
        };

        log::info!("Response writer running for session {session_id}");

        // Drain the per-session response queue. The queue closes when every sender is dropped:
        // the session's worker (on close/disconnect) and any in-flight readback tasks.
        while let Some(response) = receiver.recv().await {
            let bytes = match rmp_serde::to_vec(&response) {
                Ok(b) => b,
                Err(err) => {
                    log::error!(
                        "Failed to encode response for request {:?}: {err:?}",
                        response.id
                    );
                    continue;
                }
            };
            if let Err(err) = self.socket.send(Message::new(bytes.into())).await {
                log::warn!("Response send failed for session {session_id}: {err:?}; closing writer");
                return;
            }
        }

        log::info!("Response writer for session {session_id} exited (queue closed)");
    }

    /// Read the init handshake and reply with the selected device's settings, returning the
    /// session this responder serves (or `None` if the handshake failed and the stream should
    /// be dropped).
    async fn handshake(&mut self) -> Option<(SessionId, u32)> {
        let msg = match self.socket.recv().await {
            Ok(Some(m)) => m,
            Ok(None) => {
                log::info!("Response stream closed before init handshake");
                return None;
            }
            Err(err) => {
                log::warn!("Response stream error during init handshake: {err:?}");
                return None;
            }
        };

        let (session_id, device_index) = match parse_init_handshake(&msg.data) {
            Ok(pair) => pair,
            Err(err) => {
                log::error!("{err}; closing stream");
                return None;
            }
        };

        log::info!("Init responder for session {session_id} (device {device_index})");

        // Reply with the selected device's default settings — the client uses these to fill in
        // `RemoteDevice::defaults` so it can resolve op dtypes without an extra RTT — and the
        // device count, so it can enumerate every device behind the address.
        let settings = self.service.device_settings(device_index);
        let device_count = self.service.device_count();
        let init_response = TaskResponse {
            content: TaskResponseContent::Init(settings, device_count),
            // Placeholder id for the handshake; the client reads this response inline before
            // the response-demux task starts, so it never goes through the pending-callback
            // map.
            id: 0,
        };
        let bytes = match rmp_serde::to_vec(&init_response) {
            Ok(b) => b,
            Err(err) => {
                log::error!("Failed to encode Init response: {err:?}");
                return None;
            }
        };
        if let Err(err) = self.socket.send(Message::new(bytes.into())).await {
            log::error!("Failed to send Init response for session {session_id}: {err:?}");
            return None;
        }

        Some((session_id, device_index))
    }
}

/// The `/request` connection: decode each incoming task batch and forward its compute tasks to
/// the bound session's worker, tearing the session down when the stream ends.
pub(crate) struct SocketRequest<S, C> {
    socket: C,
    dispatch: RequestDispatch<S>,
}

impl<S: RequestService, C: CommunicationChannel> SocketRequest<S, C> {
    pub(crate) fn new(service: Arc<S>, socket: C) -> Self {
        Self {
            socket,
            dispatch: RequestDispatch::new(service),
        }
    }

    pub(crate) async fn run(mut self) {
        log::info!("[Request Handler] On new connection.");

        loop {
            let msg = match self.socket.recv().await {
                Ok(Some(m)) => m,
                Ok(None) => {
                    log::info!("Request stream closed");
                    break;
                }
                Err(err) => {
                    log::warn!("Request stream error: {err:?}; closing");
                    break;
                }
            };

            let tasks: Vec<Task> = match rmp_serde::from_slice(&msg.data) {
                Ok(t) => t,
                Err(err) => {
                    log::error!("Failed to decode request batch: {err:?}; closing stream");
                    break;
                }
            };

            if self.dispatch.process_batch(tasks).await {
                break;
            }
        }

        self.dispatch.cleanup().await;
    }
}

/// The socket-free half of the request handler: everything that decides what to do with the
/// decoded tasks and talks to the session layer, with no dependency on the wire. Kept separate
/// from the socket so it can be driven directly in tests.
struct RequestDispatch<S> {
    service: Arc<S>,
    state: RequestState,
}

impl<S: RequestService> RequestDispatch<S> {
    fn new(service: Arc<S>) -> Self {
        Self {
            service,
            state: RequestState::default(),
        }
    }

    /// Process one decoded batch in order. Each websocket frame may carry a batch of tasks (the
    /// client buffers fire-and-forget tasks before sending). Returns `true` if the connection
    /// should close — a `Close` short-circuits the rest of the batch, matching the client side
    /// (once it sends `Close`, it sends nothing after).
    async fn process_batch(&mut self, tasks: Vec<Task>) -> bool {
        for task in tasks {
            match self.state.classify(task) {
                Dispatch::Continue => {}
                Dispatch::Forward {
                    session_id,
                    compute,
                } => {
                    if !self.forward(session_id, compute).await {
                        return true;
                    }
                }
                Dispatch::Close(id) => {
                    log::info!("Close requested for session {id}");
                    self.service.close(id).await;
                    return true;
                }
                Dispatch::Abort => {
                    log::error!("Compute task received before session Init; closing request stream");
                    return true;
                }
            }
        }
        false
    }

    /// Forward a compute task to the session worker, resolving and caching the worker channel
    /// on first use. Returns `false` if the worker is gone (the connection should close), which
    /// shouldn't happen while the session is live.
    ///
    /// A full channel applies async backpressure here — the `send` await yields and we stop
    /// reading the socket — rather than blocking a runtime thread.
    async fn forward(&mut self, session_id: SessionId, compute: ComputeTask) -> bool {
        if self.state.task_sender.is_none() {
            self.state.task_sender = Some(
                self.service
                    .session_task_sender(session_id, self.state.device_index)
                    .await,
            );
        }
        let sender = self.state.task_sender.as_ref().unwrap();

        if sender.send(compute).await.is_err() {
            log::error!("Session {session_id} worker channel closed; closing request stream");
            return false;
        }
        true
    }

    /// Tear the session down once the request loop ends. The worker, runner, and response
    /// queue are released because closing the session — and dropping our cached task sender on
    /// return — closes the worker's channel, so it flushes and exits, which in turn closes the
    /// response writer's queue.
    async fn cleanup(&self) {
        match self.state.session_id {
            Some(id) if !self.state.session_closed => {
                // The stream ended without an explicit `Close` (client crash, dropped socket,
                // decode/stream error). Close here so nothing leaks.
                log::info!("Request stream for session {id} ended without Close; cleaning up");
                self.service.close(id).await;
            }
            Some(id) => log::info!("Closing session {id}"),
            None => log::info!("Closing session (no id info)"),
        }
    }
}

/// Which session a request connection is bound to, plus the channel to that session's worker.
///
/// A single connection serves one session at a time but may be re-bound to a new session by a
/// later `Task::Init` (the client never does this today, but the protocol allows it), which is
/// why the session id is optional and the worker channel is re-resolved on rebind.
#[derive(Default)]
struct RequestState {
    /// The session this connection is currently bound to, set by `Task::Init`.
    session_id: Option<SessionId>,
    /// The device index the bound session is pinned to, fixed by `Task::Init`.
    device_index: u32,
    /// Whether an explicit `Task::Close` already tore the session down, so the final cleanup
    /// doesn't double-close.
    session_closed: bool,
    /// The bound session's worker channel, resolved on the first compute task and reused for
    /// the rest of the connection. Cleared on rebind so the new session is resolved afresh.
    task_sender: Option<mpsc::Sender<ComputeTask>>,
}

/// What the request loop should do with a single decoded [`Task`].
///
/// `Forward` carries a whole `ComputeTask`, so it dwarfs the other variants — but a `Dispatch`
/// is returned by `classify` and matched immediately, never stored or collected, so the size
/// disparity is free. Boxing the task instead would add a heap allocation on every op.
#[allow(clippy::large_enum_variant)]
enum Dispatch {
    /// State-only task (an `Init` rebind); move on to the next task in the batch.
    Continue,
    /// Forward this compute task to the bound session's worker.
    Forward {
        session_id: SessionId,
        compute: ComputeTask,
    },
    /// Explicit `Close` for `session_id`: tear the session down and end the connection.
    Close(SessionId),
    /// Protocol violation (a compute task before any `Init`): end the connection.
    Abort,
}

impl RequestState {
    /// Advance the connection state by one task and report what the loop should do with it.
    /// Pure: it touches no socket, service, or backend, so the whole request dispatch can be
    /// exercised in isolation.
    fn classify(&mut self, task: Task) -> Dispatch {
        match task {
            Task::Init(id, index) => {
                self.session_id = Some(id);
                self.device_index = index;
                // Re-resolve the worker channel for the (re)bound session on the next compute
                // task.
                self.task_sender = None;
                Dispatch::Continue
            }
            Task::Close(id) => {
                self.session_closed = true;
                Dispatch::Close(id)
            }
            Task::Compute(compute) => match self.session_id {
                Some(session_id) => Dispatch::Forward {
                    session_id,
                    compute,
                },
                None => Dispatch::Abort,
            },
        }
    }
}

/// Decode the single `Task::Init` a fresh response (or request) socket opens with.
///
/// The handshake frame must hold exactly one task and it must be an `Init`; anything else is a
/// protocol error.
fn parse_init_handshake(bytes: &[u8]) -> Result<(SessionId, u32), String> {
    let mut tasks = rmp_serde::from_slice::<Vec<Task>>(bytes)
        .map_err(|err| format!("Failed to decode init handshake: {err:?}"))?;

    match tasks.pop() {
        Some(Task::Init(id, device_index)) if tasks.is_empty() => Ok((id, device_index)),
        other => Err(format!(
            "Init handshake expected a single Task::Init, got {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn encode(tasks: &[Task]) -> Vec<u8> {
        rmp_serde::to_vec(tasks).unwrap()
    }

    fn block_on<F: Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(fut)
    }

    /// A fake [`RequestService`] that records what it was asked to do: it hands out a single
    /// worker channel (whose receiver the test holds) and logs every `close`.
    struct FakeService {
        task_tx: mpsc::Sender<ComputeTask>,
        closed: Mutex<Vec<SessionId>>,
    }

    impl FakeService {
        fn new() -> (Arc<Self>, mpsc::Receiver<ComputeTask>) {
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

    impl RequestService for FakeService {
        async fn session_task_sender(
            &self,
            _session_id: SessionId,
            _device_index: u32,
        ) -> mpsc::Sender<ComputeTask> {
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

    // --- parse_init_handshake -------------------------------------------------------------

    #[test]
    fn handshake_accepts_a_single_init() {
        let id = SessionId::new();
        let bytes = encode(&[Task::Init(id, 3)]);
        assert_eq!(parse_init_handshake(&bytes).unwrap(), (id, 3));
    }

    #[test]
    fn handshake_rejects_an_empty_batch() {
        assert!(parse_init_handshake(&encode(&[])).is_err());
    }

    #[test]
    fn handshake_rejects_more_than_one_task() {
        let id = SessionId::new();
        let bytes = encode(&[Task::Init(id, 0), Task::Close(id)]);
        assert!(parse_init_handshake(&bytes).is_err());
    }

    #[test]
    fn handshake_rejects_a_non_init() {
        let id = SessionId::new();
        assert!(parse_init_handshake(&encode(&[Task::Close(id)])).is_err());
    }

    #[test]
    fn handshake_rejects_garbage() {
        assert!(parse_init_handshake(&[0xff, 0x00, 0x13]).is_err());
    }

    // --- RequestState::classify (pure) ----------------------------------------------------

    #[test]
    fn classify_init_binds_session_and_resets_worker_channel() {
        let mut state = RequestState::default();
        // A worker channel left over from a previous binding must be dropped on rebind.
        state.task_sender = Some(mpsc::channel(1).0);

        let id = SessionId::new();
        let dispatch = state.classify(Task::Init(id, 5));

        assert!(matches!(dispatch, Dispatch::Continue));
        assert_eq!(state.session_id, Some(id));
        assert_eq!(state.device_index, 5);
        assert!(state.task_sender.is_none());
        assert!(!state.session_closed);
    }

    #[test]
    fn classify_compute_before_init_aborts() {
        let mut state = RequestState::default();
        let dispatch = state.classify(Task::Compute(ComputeTask::Seed(1)));
        assert!(matches!(dispatch, Dispatch::Abort));
    }

    #[test]
    fn classify_compute_after_init_forwards_to_bound_session() {
        let mut state = RequestState::default();
        let id = SessionId::new();
        state.classify(Task::Init(id, 0));

        match state.classify(Task::Compute(ComputeTask::Seed(7))) {
            Dispatch::Forward {
                session_id,
                compute,
            } => {
                assert_eq!(session_id, id);
                assert!(matches!(compute, ComputeTask::Seed(7)));
            }
            _ => panic!("expected a forward to the bound session"),
        }
    }

    #[test]
    fn classify_close_marks_the_session_closed() {
        let mut state = RequestState::default();
        let id = SessionId::new();
        state.classify(Task::Init(id, 0));

        let dispatch = state.classify(Task::Close(id));

        assert!(matches!(dispatch, Dispatch::Close(closed) if closed == id));
        assert!(state.session_closed);
    }

    // --- RequestDispatch (against the fake service) ---------------------------------------

    #[test]
    fn dispatch_forwards_pre_close_computes_and_stops_at_close() {
        block_on(async {
            let (service, mut task_rx) = FakeService::new();
            let mut dispatch = RequestDispatch::new(service.clone());
            let id = SessionId::new();

            let stop = dispatch
                .process_batch(vec![
                    Task::Init(id, 0),
                    Task::Compute(ComputeTask::Seed(1)),
                    Task::Compute(ComputeTask::Seed(2)),
                    Task::Close(id),
                    // After `Close` the batch short-circuits: this must never be forwarded.
                    Task::Compute(ComputeTask::Seed(3)),
                ])
                .await;

            assert!(stop, "an explicit Close ends the connection");
            assert_eq!(service.closed(), vec![id]);

            let forwarded = drain(&mut task_rx);
            assert_eq!(forwarded.len(), 2, "only the two pre-Close computes forward");
            assert!(matches!(forwarded[0], ComputeTask::Seed(1)));
            assert!(matches!(forwarded[1], ComputeTask::Seed(2)));
        });
    }

    #[test]
    fn dispatch_aborts_on_compute_before_init() {
        block_on(async {
            let (service, _rx) = FakeService::new();
            let mut dispatch = RequestDispatch::new(service.clone());

            let stop = dispatch
                .process_batch(vec![Task::Compute(ComputeTask::Seed(1))])
                .await;

            assert!(stop, "a compute before any Init ends the connection");
            assert!(service.closed().is_empty(), "nothing to close yet");
        });
    }

    #[test]
    fn dispatch_cleanup_closes_a_session_that_ended_without_close() {
        block_on(async {
            let (service, _rx) = FakeService::new();
            let mut dispatch = RequestDispatch::new(service.clone());
            let id = SessionId::new();

            // The batch never sends `Close` — simulate an abrupt disconnect afterwards.
            dispatch
                .process_batch(vec![Task::Init(id, 0), Task::Compute(ComputeTask::Seed(1))])
                .await;
            dispatch.cleanup().await;

            assert_eq!(service.closed(), vec![id]);
        });
    }

    #[test]
    fn dispatch_cleanup_does_not_double_close_after_explicit_close() {
        block_on(async {
            let (service, _rx) = FakeService::new();
            let mut dispatch = RequestDispatch::new(service.clone());
            let id = SessionId::new();

            dispatch
                .process_batch(vec![Task::Init(id, 0), Task::Close(id)])
                .await;
            dispatch.cleanup().await;

            assert_eq!(service.closed(), vec![id], "closed exactly once");
        });
    }
}
