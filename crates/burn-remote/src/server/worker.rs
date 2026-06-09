//! Per-session handler and its per-stream worker threads.
//!
//! A [`SessionHandler`] owns everything that is constant for the lifetime of a session — the
//! session's [`TensorInterpreter`] (with its own [`HandleContainer`](burn_ir::HandleContainer)),
//! the cross-server / same-host comm services, the result sender, the runtime handle and the
//! session id — and exposes a single [`process_task`](SessionHandler::process_task) method that
//! runs one task against that state.
//!
//! Tasks do not all run on one thread. The submit handler forwards every task for the session to
//! a single inbound channel; a **dispatcher** thread drains that channel and routes each task —
//! by its [`StreamId`] — to a **per-stream worker thread**, spawning one lazily the first time a
//! stream is seen. Each stream worker drives its own tasks in FIFO order, so per-stream ordering
//! (the only ordering the protocol promises) is preserved, while *independent* streams run
//! concurrently and never head-of-line block one another.
//!
//! ## Why per-stream threads, not one session thread
//!
//! Some tasks are **synchronously blocking**: a same-host transfer's
//! [`RegisterTensorLocal`](crate::shared::Task::RegisterTensorLocal) waits on the rendezvous
//! (`local_comm.take`) for its source session to expose the primitive, and a collective op
//! (all-reduce / sync-collective) parks until *every* participating device reaches the barrier.
//! If a single session thread processed every stream in FIFO order, such a blocking wait would
//! stall every *later* task on the session — including an
//! [`ExposeTensorLocal`](crate::shared::Task::ExposeTensorLocal) that another session is waiting
//! on. Two devices transferring in opposite directions at once would then deadlock: each
//! session's thread blocked on a `take` whose matching `expose` is queued behind the *other*
//! session's blocked `take`.
//!
//! Splitting a session into one OS thread per stream removes that false dependency: a blocking
//! `take` (or barrier) parks only its own stream, and the expose the peer needs — on a different
//! stream — keeps flowing. A dedicated OS thread per stream (rather than a tokio task) also keeps
//! the blocking wait off the shared runtime's worker threads, so a parked stream can't starve the
//! runtime.

use std::collections::HashMap;
use std::sync::Arc;

use burn_communication::{Protocol, external_comm::ExternalCommService};
use burn_ir::BackendIr;
use burn_router::{RouterClient, TensorInterpreter};
use burn_std::id::StreamId;
use tokio::{runtime::Handle, sync::mpsc};

use crate::server::local_comm::LocalCommService;
use crate::shared::{RequestId, SessionId, Task, TaskResponse, TaskResponseContent};

/// Capacity of the per-session inbound channel feeding the dispatcher.
///
/// The submit handler forwards with an `await`ing send, so a full channel applies async
/// backpressure (the submit task yields and stops reading the socket) rather than blocking an OS
/// thread or growing memory without bound. The dispatcher only *routes* tasks (a cheap
/// non-blocking send onto a per-stream channel), so it drains this quickly and the channel fills
/// only under a genuine flood, not because some stream is parked on a blocking wait.
const TASK_CHANNEL_CAPACITY: usize = 64;

/// Everything constant for the lifetime of a session.
///
/// Shared (via [`Arc`]) by the dispatcher and every per-stream worker thread, so they all run
/// tasks against the same interpreter and comm services. The interpreter is itself
/// `Arc<Mutex<…>>` internally, so concurrent stream workers mutating handles are serialized at
/// the handle container, not here.
pub(crate) struct SessionHandler<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    session_id: SessionId,
    runner: TensorInterpreter<B>,
    response_sender: mpsc::Sender<TaskResponse>,
    external_comm: Arc<ExternalCommService<B, P>>,
    local_comm: Arc<LocalCommService<B>>,
    /// Captured at construction so each per-stream thread can `block_on` the runtime that owns the
    /// IO driver — needed for the async parts of a task and for detaching readbacks with
    /// `tokio::spawn`.
    handle: Handle,
}

impl<B, P> SessionHandler<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Create a session's handler and spawn its dispatcher thread, returning the inbound task
    /// sender the submit handler forwards to.
    ///
    /// Must be called from within the tokio runtime: it captures [`Handle::current`] for the
    /// dispatcher and the per-stream workers it spawns.
    ///
    /// The returned [`mpsc::Sender`] is cloned once per submit connection. The dispatcher (and the
    /// whole session) ends when every clone is dropped — clean session close or submit-stream
    /// disconnect — at which point the per-stream channels close, their threads drain and exit,
    /// and the last handler reference dropped flushes the runner (see [`Drop`]).
    pub(crate) fn spawn(
        session_id: SessionId,
        runner: TensorInterpreter<B>,
        response_sender: mpsc::Sender<TaskResponse>,
        external_comm: Arc<ExternalCommService<B, P>>,
        local_comm: Arc<LocalCommService<B>>,
    ) -> mpsc::Sender<Task> {
        let handle = Handle::current();
        let handler = Arc::new(Self {
            session_id,
            runner,
            response_sender,
            external_comm,
            local_comm,
            handle: handle.clone(),
        });

        let (sender, receiver) = mpsc::channel(TASK_CHANNEL_CAPACITY);

        // A plain detached OS thread: it owns the per-stream registry and ends itself when the
        // inbound channel closes, so there is nothing to join.
        std::thread::Builder::new()
            .name(format!("burn-remote-session-{session_id}-dispatch"))
            .spawn(move || handler.dispatch_loop(receiver))
            .expect("Failed to spawn session dispatcher thread");

        sender
    }

    /// Drain the inbound channel, routing each task to its stream's worker (spawned lazily).
    fn dispatch_loop(self: Arc<Self>, mut receiver: mpsc::Receiver<Task>) {
        let session_id = self.session_id;
        log::info!(
            "New session dispatcher: {} {:?}",
            session_id,
            std::thread::current().id()
        );

        // The per-stream senders live here, owned by the dispatcher alone — *not* inside the
        // shared `SessionHandler`. That's deliberate: if the handler held them, the stream
        // threads (which hold an `Arc<SessionHandler>`) would keep their own channels alive and
        // never exit. Owning them here means that when the inbound channel closes and this map is
        // dropped, every stream channel closes, its thread drains and exits, and the handler is
        // finally released.
        let mut streams: HashMap<StreamId, mpsc::UnboundedSender<Task>> = HashMap::new();
        let mut processed: u64 = 0;

        self.handle.clone().block_on(async {
            while let Some(task) = receiver.recv().await {
                processed += 1;
                if processed <= 3 || processed.is_multiple_of(200) {
                    log::info!("Session {session_id} dispatcher: routed {processed} tasks");
                }
                let stream_id = task.stream_id_or_default();
                let sender = streams
                    .entry(stream_id)
                    .or_insert_with(|| self.spawn_stream_worker(stream_id));

                // An unbounded send never blocks, so a stream parked on a blocking `take`/barrier
                // can't stall the dispatcher (and thus other streams). The client self-throttles a
                // single stream by blocking on its own reads/syncs, so this can't grow without
                // bound for a stream that is actually making progress.
                if sender.send(task).is_err() {
                    log::error!(
                        "Session {session_id} stream {stream_id:?} worker gone; dropping task"
                    );
                    streams.remove(&stream_id);
                }
            }
        });

        log::info!("Session {session_id} dispatcher: drained after {processed} tasks");
        // Dropping `streams` closes every per-stream channel, so the workers drain and exit.
    }

    /// Spawn the dedicated OS thread that drives one stream's tasks in FIFO order.
    fn spawn_stream_worker(self: &Arc<Self>, stream_id: StreamId) -> mpsc::UnboundedSender<Task> {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let handler = self.clone();
        let handle = self.handle.clone();
        let session_id = self.session_id;

        std::thread::Builder::new()
            .name(format!("burn-remote-session-{session_id}-stream-{stream_id:?}"))
            .spawn(move || {
                // Drive this stream's tasks to completion on this thread. The synchronous parts
                // (collective barriers, op registration, the `local_comm.take` wait,
                // `runner.sync()`) block only this thread; the async parts are driven by
                // `block_on`, and detached readbacks spawned inside run on the shared runtime, so
                // they make progress while this thread is parked.
                handle.block_on(async move {
                    while let Some(task) = receiver.recv().await {
                        if let Err(err) = handler.process_task(task).await {
                            // One task failing doesn't tear down the stream: read/sync/dtype
                            // failures surface to the client through their response,
                            // fire-and-forget failures are logged here, and the worker keeps
                            // processing subsequent tasks.
                            log::error!(
                                "Task on session {session_id} stream {stream_id:?} failed: {err}"
                            );
                        }
                    }
                });
            })
            .expect("Failed to spawn session stream worker thread");

        sender
    }

    /// Execute a single [`Task`] against this session's state.
    ///
    /// Sync work is wrapped in [`StreamId::executes`] so the runner's thread-local stream id
    /// matches the one the client assigned to this op. Response-producing tasks carry their own
    /// [`RequestId`] for routing the response back to the right pending callback on the client.
    /// Async work (data-service transfers, `read_tensor_async`) runs without a stream context —
    /// the relevant stream id is captured into the future at construction time via `executes`.
    async fn process_task(&self, task: Task) -> Result<(), String> {
        let runner = &self.runner;
        match task {
            Task::RegisterOperation(stream_id, op) => {
                stream_id.executes(|| runner.register_op(op));
                Ok(())
            }
            Task::RegisterTensor(stream_id, id, data) => {
                stream_id.executes(|| runner.register_tensor_data_id(id, data));
                Ok(())
            }
            Task::RegisterTensorRemote(stream_id, remote, new_id) => {
                log::info!(
                    "Registering remote tensor (transfer {:?} from {:?})",
                    remote.transfer_id,
                    remote.address,
                );
                let data = self
                    .external_comm
                    .download_tensor(remote.address.clone(), remote.transfer_id)
                    .await
                    .ok_or_else(|| {
                        format!(
                            "Failed to download tensor for transfer {:?} from {:?}",
                            remote.transfer_id, remote.address,
                        )
                    })?;
                // Register on the client stream that will consume `new_id`, carried over the
                // wire — not the arbitrary tokio worker running this task.
                stream_id.executes(|| runner.register_tensor_data_id(new_id, data));
                Ok(())
            }
            Task::ExposeTensorLocal {
                stream_id,
                tensor,
                transfer_id,
            } => {
                // Source side of a same-host transfer. Grab the device-resident primitive
                // (no host readback) and park it in the registry for the target session to
                // pick up. Runs in order on this stream's worker, so it is ordered after the op
                // that produced `tensor` — the handle is guaranteed present. Read it back on the
                // client stream that produced it, carried over the wire.
                let kind = stream_id.executes(|| runner.get_tensor(&tensor));
                self.local_comm.expose(transfer_id, kind).await;
                Ok(())
            }
            Task::RegisterTensorLocal {
                stream_id,
                transfer_id,
                new_id,
            } => {
                // Target side of a same-host transfer. Wait for the source to expose the
                // primitive, then move it onto this session's device and register it. Awaited
                // in order so subsequent ops on this stream that consume `new_id` see it
                // registered first — same ordering contract as `RegisterTensorRemote`. The wait
                // blocks only this stream's worker, not the source session's, and not other
                // streams on this session.
                let kind = self.local_comm.take(transfer_id).await;
                stream_id.executes(|| runner.register_tensor_to_device(new_id, kind));
                Ok(())
            }
            Task::ExposeTensorRemote {
                stream_id,
                tensor,
                count,
                transfer_id,
            } => {
                log::info!("Exposing tensor (transfer {transfer_id:?})");
                // Same shape as `ReadTensor`: the sync part of `read_tensor_async` runs in order
                // to preserve stream ordering, but the readback + expose are detached so a
                // cross-server hand-off doesn't stall this stream's op registration on a
                // GPU→host copy. A target that downloads before the expose lands simply blocks on
                // the data service's `new_tensor_notify`, so there is no race.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                let external_comm = self.external_comm.clone();
                tokio::spawn(async move {
                    match fut.await {
                        Ok(data) => {
                            external_comm.expose_data(data, count, transfer_id).await;
                        }
                        Err(e) => {
                            log::error!(
                                "read_tensor_async for transfer {transfer_id:?} failed: {e:?}"
                            );
                        }
                    }
                });
                Ok(())
            }
            Task::Seed(seed) => {
                runner.seed(seed);
                Ok(())
            }
            Task::ReadTensor(request_id, stream_id, tensor) => {
                // `read_tensor_async` is sync at construction — it locks the context and
                // captures the tensor's position in the command stream — and returns a future
                // for the actual host readback. Run the sync part in order (so ordering vs. later
                // ops is preserved), then detach the readback await onto its own task. Awaiting it
                // here would stall the stream on the GPU→host copy and stop us registering
                // subsequent ops, draining the device queue into a bubble. The client demuxes
                // responses by request id, so out-of-order completion is fine.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                let sender = self.response_sender.clone();
                tokio::spawn(async move {
                    let data = fut.await;
                    if sender
                        .send(TaskResponse {
                            content: TaskResponseContent::ReadTensor(data),
                            id: request_id,
                        })
                        .await
                        .is_err()
                    {
                        log::warn!(
                            "Response receiver dropped before read for request {request_id} could be sent"
                        );
                    }
                });
                Ok(())
            }
            Task::SyncBackend(request_id, stream_id) => {
                let res = stream_id.executes(|| runner.sync());
                self.send_response(request_id, TaskResponseContent::SyncBackend(res))
                    .await
            }
            Task::DTypeUsage(request_id, dtype) => {
                let res = runner.dtype_usage(dtype);
                self.send_response(request_id, TaskResponseContent::DTypeUsage(res))
                    .await
            }
        }
    }

    async fn send_response(
        &self,
        request_id: RequestId,
        content: TaskResponseContent,
    ) -> Result<(), String> {
        self.response_sender
            .send(TaskResponse {
                content,
                id: request_id,
            })
            .await
            .map_err(|_| {
                format!(
                    "Response receiver dropped before result for request {request_id} could be sent"
                )
            })
    }
}

impl<B, P> Drop for SessionHandler<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Flush outstanding backend work before the session's tensors are freed.
    ///
    /// This runs once, when the last reference to the handler is dropped — i.e. after the
    /// dispatcher and every per-stream worker have exited (each holds an `Arc<Self>`). Flushing
    /// here means the session's tensors aren't freed with GPU work still queued. Dropping the
    /// `response_sender` (a field) afterwards closes the fetch writer's queue, ending its task
    /// too.
    fn drop(&mut self) {
        let session_id = self.session_id;
        log::info!("Session {session_id} handler draining and exiting");
        if let Err(err) = self.runner.sync() {
            log::warn!("runner.sync() at session {session_id} close failed: {err:?}");
        }
        let device = self.runner.device();
        if let Err(err) = B::sync(&device) {
            log::warn!("B::sync(device) at session {session_id} close failed: {err:?}");
        }
    }
}
