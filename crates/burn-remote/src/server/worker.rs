//! Per-session worker thread.
//!
//! Each session owns one dedicated OS thread that runs all of its tasks. The websocket submit
//! handler does no work itself: it decodes the incoming message batch and forwards each
//! [`Task`] to the session's worker over a bounded channel. The worker owns the session's
//! [`TensorInterpreter`] and processes tasks in FIFO order, preserving per-session ordering
//! without any extra locking.
//!
//! Why a dedicated OS thread instead of running tasks on the tokio submit task? Some tasks
//! are **synchronously blocking** — most importantly a collective op (all-reduce /
//! sync-collective), which parks the calling thread until *every* participating device
//! reaches the same barrier (`register_op` → `B::sync_collective` →
//! `DistributedSyncClient::submit_sync_collective` → a blocking `rx.recv()`). Running that on
//! a tokio worker ties up a runtime thread; once more devices are blocked than the runtime
//! has worker threads, the remaining devices can never be scheduled to reach the barrier and
//! it deadlocks. Giving each session its own OS thread keeps that blocking off the shared
//! runtime, so a barrier on one device can't stall another — the submit loops (tokio tasks)
//! keep forwarding and every session's worker reaches the barrier independently.

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
/// backpressure (the submit task yields and stops reading the socket) rather than blocking
/// an OS thread or growing memory without bound. Sized so a burst of fire-and-forget ops
/// doesn't stall the submit loop while the worker is mid-task.
const TASK_CHANNEL_CAPACITY: usize = 64;

/// Handle to a session's worker thread, held by the session map and cloned once per submit
/// connection. The worker thread runs until every clone of its task sender is dropped (clean
/// session close or submit-stream disconnect), at which point it flushes the runner and
/// exits — so the handle is detached and there is nothing to join.
pub(crate) struct SessionWorker {
    sender: mpsc::Sender<Task>,
}

impl SessionWorker {
    /// Spawn the worker thread for a freshly created session.
    ///
    /// Must be called from within the tokio runtime: it captures [`Handle::current`] so the
    /// worker can drive the async parts of a task (cross-server / same-host transfers,
    /// tensor readbacks, response sends) and detach readbacks with `tokio::spawn`.
    pub(crate) fn new<B, P>(
        session_id: SessionId,
        runner: TensorInterpreter<B>,
        response_sender: mpsc::Sender<TaskResponse>,
        external_comm: Arc<ExternalCommService<B, P>>,
        local_comm: Arc<LocalCommService<B>>,
    ) -> Self
    where
        B: BackendIr,
        P: Protocol,
    {
        let (sender, receiver) = mpsc::channel(TASK_CHANNEL_CAPACITY);
        let handle = Handle::current();

        // A plain OS thread, detached: it owns the runner and ends itself when the task
        // channel closes, so there is nothing to join.
        std::thread::Builder::new()
            .name(format!("burn-remote-session-{session_id}"))
            .spawn(move || {
                worker_loop(
                    handle,
                    receiver,
                    session_id,
                    runner,
                    response_sender,
                    external_comm,
                    local_comm,
                );
            })
            .expect("Failed to spawn session worker thread");

        Self { sender }
    }

    /// Clone the channel used to forward tasks to this worker.
    pub(crate) fn task_sender(&self) -> mpsc::Sender<Task> {
        self.sender.clone()
    }
}

fn worker_loop<B, P>(
    handle: Handle,
    mut receiver: mpsc::Receiver<Task>,
    session_id: SessionId,
    runner: TensorInterpreter<B>,
    response_sender: mpsc::Sender<TaskResponse>,
    external_comm: Arc<ExternalCommService<B, P>>,
    local_comm: Arc<LocalCommService<B>>,
) where
    B: BackendIr,
    P: Protocol,
{
    // Drive every task to completion on this thread. The synchronous parts (collective
    // barriers, op registration, `runner.sync()`) block only this thread; the async parts
    // are driven by `block_on`, and detached readbacks spawned inside run on the shared
    // runtime's worker threads, so they make progress while this thread is parked on a
    // barrier.
    handle.block_on(async {
        while let Some(task) = receiver.recv().await {
            if let Err(err) =
                process_task(&external_comm, &local_comm, &runner, &response_sender, task).await
            {
                // One task failing doesn't tear down the session: read/sync/dtype failures
                // surface to the client through their response, fire-and-forget failures are
                // logged here, and the worker keeps processing subsequent tasks.
                log::error!("Task on session {session_id} failed: {err}");
            }
        }
    });

    // The task channel closed: every submit connection bound to this session has gone away
    // (clean `Close` or disconnect). Flush outstanding backend work before dropping the
    // runner so the session's tensors aren't freed with GPU work still queued. Dropping the
    // runner and `response_sender` afterwards closes the fetch writer's queue, ending its
    // task too.
    log::info!("Session {session_id} worker draining and exiting");
    if let Err(err) = runner.sync() {
        log::warn!("runner.sync() at session {session_id} close failed: {err:?}");
    }
    let device = runner.device();
    if let Err(err) = B::sync(&device) {
        log::warn!("B::sync(device) at session {session_id} close failed: {err:?}");
    }
}

/// Execute a single [`Task`] on the worker thread.
///
/// Sync work is wrapped in [`StreamId::executes`](burn_std::id::StreamId::executes) so the runner's thread-local stream
/// id matches the one the client assigned to this op. Response-producing tasks carry their
/// own [`RequestId`] for routing the response back to the right pending callback on the
/// client. Async work (data-service transfers, `read_tensor_async`) runs without a stream
/// context — the relevant stream id is captured into the future at construction time via
/// `executes`.
async fn process_task<B, P>(
    external_comm: &Arc<ExternalCommService<B, P>>,
    local_comm: &Arc<LocalCommService<B>>,
    runner: &TensorInterpreter<B>,
    sender: &mpsc::Sender<TaskResponse>,
    task: Task,
) -> Result<(), String>
where
    B: BackendIr,
    P: Protocol,
{
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
            let data = external_comm
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
            local_comm.expose(transfer_id, kind).await;
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
            let kind = local_comm.take(transfer_id).await;
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
            // cross-server hand-off doesn't stall this session's op registration on a
            // GPU→host copy. A target that downloads before the expose lands simply blocks on
            // the data service's `new_tensor_notify`, so there is no race.
            let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
            let external_comm = external_comm.clone();
            tokio::spawn(async move {
                match fut.await {
                    Ok(data) => {
                        external_comm.expose_data(data, count, transfer_id).await;
                    }
                    Err(e) => {
                        log::error!("read_tensor_async for transfer {transfer_id:?} failed: {e:?}");
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
            let sender = sender.clone();
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
            send_response(sender, request_id, TaskResponseContent::SyncBackend(res)).await
        }
        Task::DTypeUsage(request_id, dtype) => {
            let res = runner.dtype_usage(dtype);
            send_response(sender, request_id, TaskResponseContent::DTypeUsage(res)).await
        }
    }
}

async fn send_response(
    sender: &mpsc::Sender<TaskResponse>,
    request_id: RequestId,
    content: TaskResponseContent,
) -> Result<(), String> {
    sender
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
