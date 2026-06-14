//! Per-session handler and its worker thread.
//!
//! A [`SessionHandler`] owns everything that is constant for the lifetime of a session — the
//! session's [`TensorInterpreter`] (with its own [`HandleContainer`](burn_ir::HandleContainer)),
//! the cross-server / same-host comm services, the result sender, the runtime handle and the
//! session id — and exposes a single [`process_task`](SessionHandler::process_task) method that
//! runs one task against that state.
//!
//! The submit handler does no work itself: it decodes the incoming message batch and forwards
//! each [`Task`] to the session's worker over a bounded channel. The worker drives the session's
//! tasks in **global submission order** (a single FIFO), preserving every ordering the protocol
//! relies on — including *cross-stream* ones. A tensor produced on one client stream (say a
//! dataloader thread) and consumed on another (the main thread) is only safe because the producer
//! task is applied before the consumer task; the interpreter looks up input handles eagerly and
//! panics if one is missing, so the order the client submitted in must be preserved verbatim.
//! Per-stream parallelism would break exactly this, so the stream id on a task is used only to set
//! the backend's thread-local stream (via [`StreamId::executes`]), not to reorder work.
//!
//! ## Why a dedicated OS thread, not a tokio task
//!
//! Some tasks are **synchronously blocking** — a same-host transfer's
//! [`RegisterTensorLocal`](crate::shared::Task::RegisterTensorLocal) waits on the rendezvous
//! (`local_comm.take`) for its source session to expose the primitive, and a collective op
//! (all-reduce / sync-collective) parks until *every* participating device reaches the barrier.
//! Running those on a shared tokio worker would tie up a runtime thread; once more devices are
//! blocked than the runtime has workers, the remaining devices can never be scheduled to reach the
//! barrier and it deadlocks. Giving each session its own OS thread keeps that blocking off the
//! shared runtime, so a barrier (or rendezvous) on one session can't stall another session's
//! worker or a runtime thread.

use std::sync::Arc;

use burn_communication::{Protocol, external_comm::ExternalCommService};
use burn_ir::BackendIr;
use burn_router::{RouterClient, TensorInterpreter};
use tokio::{runtime::Handle, sync::mpsc};

use crate::server::local_comm::LocalCommService;
use crate::shared::{RequestId, SessionId, Task, TaskResponse, TaskResponseContent};

/// Capacity of the per-session task channel feeding the worker thread.
///
/// The submit handler forwards with an `await`ing send, so a full channel applies async
/// backpressure (the submit task yields and stops reading the socket) rather than blocking an OS
/// thread or growing memory without bound. Sized so a burst of fire-and-forget ops doesn't stall
/// the submit loop while the worker is mid-task.
const TASK_CHANNEL_CAPACITY: usize = 64;

/// Everything constant for the lifetime of a session.
///
/// Owned by the session's worker thread, which runs every task against this one interpreter and
/// these comm services. Held behind an [`Arc`] only so detached readback tasks can be spawned with
/// a clone of the result sender; the interpreter itself is single-owner here.
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
}

impl<B, P> SessionHandler<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Create a session's handler and spawn its worker thread, returning the inbound task sender
    /// the submit handler forwards to.
    ///
    /// Must be called from within the tokio runtime: it captures [`Handle::current`] so the worker
    /// can drive the async parts of a task (cross-server / same-host transfers, tensor readbacks,
    /// response sends) and detach readbacks with `tokio::spawn`.
    ///
    /// The returned [`mpsc::Sender`] is cloned once per submit connection. The worker runs until
    /// every clone is dropped — clean session close or submit-stream disconnect — at which point it
    /// flushes the runner and exits, so the handle is detached and there is nothing to join.
    pub(crate) fn spawn(
        session_id: SessionId,
        runner: TensorInterpreter<B>,
        response_sender: mpsc::Sender<TaskResponse>,
        external_comm: Arc<ExternalCommService<B, P>>,
        local_comm: Arc<LocalCommService<B>>,
    ) -> mpsc::Sender<Task> {
        let handle = Handle::current();
        let handler = SessionHandler {
            session_id,
            runner,
            response_sender,
            external_comm,
            local_comm,
        };

        let (sender, receiver) = mpsc::channel(TASK_CHANNEL_CAPACITY);

        // A plain detached OS thread: it owns the runner and ends itself when the task channel
        // closes, so there is nothing to join.
        std::thread::Builder::new()
            .name(format!("burn-remote-session-{session_id}"))
            .spawn(move || handler.worker_loop(handle, receiver))
            .expect("Failed to spawn session worker thread");

        sender
    }

    /// Drain the task channel, running each task to completion in arrival order.
    fn worker_loop(self, handle: Handle, mut receiver: mpsc::Receiver<Task>) {
        let session_id = self.session_id;

        // Drive every task to completion on this thread. The synchronous parts (collective
        // barriers, op registration, the `local_comm.take` wait, `runner.sync()`) block only this
        // thread; the async parts are driven by `block_on`, and detached readbacks spawned inside
        // run on the shared runtime's worker threads, so they make progress while this thread is
        // parked on a barrier or rendezvous.
        handle.block_on(async {
            log::debug!("Session {session_id} worker started");
            while let Some(task) = receiver.recv().await {
                if let Err(err) = self.process_task(task).await {
                    // One task failing doesn't tear down the session: read/sync/dtype failures
                    // surface to the client through their response, fire-and-forget failures are
                    // logged here, and the worker keeps processing subsequent tasks.
                    log::error!("Task on session {session_id} failed: {err}");
                }
            }

            // Reclaim any same-host transfers this session exposed that no target ever took, so a
            // half-finished transfer doesn't strand device memory in the shared registry.
            self.local_comm.purge_session(session_id).await;
        });

        // The task channel closed: every submit connection bound to this session has gone away
        // (clean `Close` or disconnect). Tear the session down in an order that actually releases
        // its memory.
        log::debug!("Session {session_id} worker draining and exiting");

        // Destructure so we control drop order explicitly. The `..` fields (response sender, comm
        // services) drop here: dropping the response sender closes the fetch writer's queue, ending
        // its task.
        let SessionHandler { runner, .. } = self;
        let device = runner.device();

        // Flush outstanding backend work before dropping the runner so the session's tensors aren't
        // freed with GPU work still queued against them.
        if let Err(err) = runner.sync() {
            log::warn!("runner.sync() at session {session_id} close failed: {err:?}");
        }
        if let Err(err) = B::sync(&device) {
            log::warn!("B::sync(device) at session {session_id} close failed: {err:?}");
        }

        // Drop the runner: this frees every tensor handle the session held back to the backend's
        // allocator. Must happen before `memory_cleanup`, otherwise the memory is still live and
        // nothing is reclaimable.
        drop(runner);

        // Hand the freed memory back instead of leaving it parked in the allocator's pool for a
        // session that no longer exists. A no-op on backends that don't pool (the `Backend`
        // default), but on cubecl backends (wgpu/cuda) this returns the session's device memory so
        // a long-lived server doesn't accumulate it across session churn.
        B::memory_cleanup(&device);
    }

    /// Execute a single [`Task`] against this session's state.
    ///
    /// Sync work is wrapped in [`StreamId::executes`](burn_std::id::StreamId::executes) so the
    /// runner's thread-local stream id matches the one the client assigned to this op.
    /// Response-producing tasks carry their own [`RequestId`] for routing the response back to the
    /// right pending callback on the client. Async work (data-service transfers,
    /// `read_tensor_async`) runs without a stream context — the relevant stream id is captured into
    /// the future at construction time via `executes`.
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
                log::trace!(
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
                // pick up. Runs in order on this session's worker, so it is ordered after the op
                // that produced `tensor` — the handle is guaranteed present. Read it back on the
                // client stream that produced it, carried over the wire.
                let kind = stream_id.executes(|| runner.get_tensor(&tensor));
                self.local_comm
                    .expose(self.session_id, transfer_id, kind)
                    .await;
                Ok(())
            }
            Task::RegisterTensorLocal {
                stream_id,
                transfer_id,
                new_id,
            } => {
                // Target side of a same-host transfer. Wait for the source to expose the
                // primitive, then move it onto this session's device and register it. Awaited
                // in order so subsequent ops on this session that consume `new_id` see it
                // registered first — same ordering contract as `RegisterTensorRemote`. The wait
                // blocks only this session's worker, not the source session's.
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
                log::trace!("Exposing tensor (transfer {transfer_id:?})");
                // Same shape as `ReadTensor`: the sync part of `read_tensor_async` runs in order
                // to preserve stream ordering, but the readback + expose are detached so a
                // cross-server hand-off doesn't stall this session's op registration on a
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
                // here would stall the worker on the GPU→host copy and stop us registering
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
