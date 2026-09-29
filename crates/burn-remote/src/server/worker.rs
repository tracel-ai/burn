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
//!
//! In the browser the worker instead runs on the JS event loop via `spawn_local`, so a browser peer
//! cannot host co-located collective / same-host-transfer participants (the single thread would
//! stall on the blocking tasks above). Ordinary compute is unaffected.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use burn_ir::{BackendIr, GraphBindings, GraphId};
use burn_router::{Graph, TensorInterpreter};
use burn_std::id::StreamId;
#[cfg(not(target_family = "wasm"))]
use std::{
    any::Any,
    panic::{self, AssertUnwindSafe},
};
#[cfg(not(target_family = "wasm"))]
use tokio::runtime::Handle;
use tokio::sync::mpsc;

use crate::server::local_comm::LocalCommService;
use crate::server::spawn::{ResponseTasks, spawn_detached};
use crate::server::transfer::TensorTransfer;
use crate::shared::{RequestId, SessionId, Task, TaskResponse, TaskResponseContent};
use crate::telemetry::{
    TelemetryEvent, TelemetryProbe, TransferPhase, TransferScope, serialized_len,
};
use burn_ir::OperationIr;

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
/// these comm services.
pub(crate) struct SessionHandler<B, T>
where
    B: BackendIr,
    T: TensorTransfer<B>,
{
    session_id: SessionId,
    runner: TensorInterpreter<B>,
    response_sender: mpsc::Sender<TaskResponse>,
    response_tasks: ResponseTasks,
    transfer: Arc<T>,
    local_comm: Arc<LocalCommService<B>>,
    /// Cache of client-registered reusable op-graphs, keyed by id (see
    /// [`Task::RegisterAndExecuteGraph`]).
    ///
    /// `Mutex` only for interior mutability under `process_task(&self)`. Each graph is held behind
    /// an [`Arc`] so a replay clones the handle out under a short lock and runs `replay_graph`
    /// *after* releasing it — the lock guards only the map, never the backend dispatch. That keeps
    /// the cache from becoming a serialization point if graph execution is ever driven from more
    /// than the current single-FIFO worker.
    graphs: Mutex<HashMap<GraphId, Graph>>,
    probe: TelemetryProbe,
}

impl<B, T> SessionHandler<B, T>
where
    B: BackendIr,
    T: TensorTransfer<B>,
{
    /// Start the session worker; the worker runs until every clone of the returned sender is
    /// dropped, then flushes and exits. Off wasm, a task that panics ends it the same way.
    pub(crate) fn spawn(
        session_id: SessionId,
        runner: TensorInterpreter<B>,
        response_sender: mpsc::Sender<TaskResponse>,
        transfer: Arc<T>,
        local_comm: Arc<LocalCommService<B>>,
        probe: TelemetryProbe,
    ) -> mpsc::Sender<Task> {
        let handler = SessionHandler {
            session_id,
            runner,
            response_sender,
            response_tasks: ResponseTasks::default(),
            transfer,
            local_comm,
            graphs: Mutex::new(HashMap::new()),
            probe,
        };
        let (sender, receiver) = mpsc::channel(TASK_CHANNEL_CAPACITY);
        handler.drive(receiver);
        sender
    }

    // Must be called from within the runtime that owns the endpoint: it captures `Handle::current`.
    #[cfg(not(target_family = "wasm"))]
    fn drive(mut self, receiver: mpsc::Receiver<Task>) {
        let handle = Handle::current();
        let session_id = self.session_id;
        std::thread::Builder::new()
            .name(format!("burn-remote-session-{session_id}"))
            .spawn(move || {
                // A panic can leave the interpreter half-applied, so the session is closed rather
                // than given another task.
                let processed = panic::catch_unwind(AssertUnwindSafe(|| {
                    handle.block_on(self.process_tasks(receiver))
                }));
                if let Err(payload) = processed {
                    self.abort_responses_after_panic(payload);
                }
                handle.block_on(self.close());
            })
            .expect("Failed to spawn session worker thread");
    }

    #[cfg(target_family = "wasm")]
    fn drive(mut self, receiver: mpsc::Receiver<Task>) {
        spawn_detached(async move {
            self.process_tasks(receiver).await;
            self.close().await;
        });
    }

    /// Abort the responses still in flight, since one may never resolve and holds the response
    /// queue open, and log why the task panicked.
    #[cfg(not(target_family = "wasm"))]
    fn abort_responses_after_panic(&mut self, payload: Box<dyn Any + Send>) {
        self.response_tasks.abort_all();
        let reason = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("no message");
        log::error!(
            "Session {} stopped because a task panicked: {reason}",
            self.session_id
        );
    }

    /// Run each task to completion in arrival order, until the task channel closes.
    async fn process_tasks(&mut self, mut receiver: mpsc::Receiver<Task>) {
        let session_id = self.session_id;

        log::debug!("Session {session_id} worker started");
        while let Some(task) = receiver.recv().await {
            if let Err(err) = self.process_task(task).await {
                // One task failing doesn't tear down the session: read/sync/dtype failures surface
                // to the client through their response, fire-and-forget failures are logged here,
                // and the worker keeps processing subsequent tasks.
                log::error!("Task on session {session_id} failed: {err}");
            }
        }
    }

    /// Tear the session down in an order that releases its memory.
    async fn close(self) {
        let session_id = self.session_id;

        // Reclaim any same-host transfers this session exposed that no target ever took, so a
        // half-finished transfer doesn't strand device memory in the shared registry.
        self.local_comm.purge_session(session_id).await;

        log::debug!("Session {session_id} worker draining and exiting");

        // Released first, so the client sees its session end even if a sync below hangs.
        let SessionHandler {
            runner,
            response_sender,
            ..
        } = self;
        drop(response_sender);
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
    async fn process_task(&mut self, task: Task) -> Result<(), String> {
        match task {
            Task::RegisterOperation(stream_id, op) => {
                // An op received individually (not as part of a cached graph) is an unfused op.
                self.emit_op(stream_id, &op);
                stream_id.executes(|| self.runner.register_op(op));
                Ok(())
            }
            Task::RegisterAndExecuteGraph {
                stream_id,
                graph_id,
                relative_graph,
                bindings,
            } => {
                self.probe.emit(|| {
                    TelemetryEvent::graph_registered(
                        self.session_id,
                        graph_id,
                        &relative_graph,
                        serialized_len(&relative_graph),
                    )
                });
                self.emit_graph_executed(graph_id, stream_id, &bindings);
                stream_id.executes(|| {
                    let graph = Graph::new(relative_graph);
                    self.graphs.lock().unwrap().insert(graph_id, graph.clone());
                    graph.replay(&mut self.runner, bindings);
                });
                Ok(())
            }
            Task::ExecuteGraph {
                stream_id,
                graph_id,
                bindings,
            } => {
                let graph = {
                    let cache = self.graphs.lock().unwrap();
                    cache
                        .get(&graph_id)
                        .cloned()
                        .ok_or_else(|| format!("Execute of unknown graph {graph_id:?}"))?
                };
                self.emit_graph_executed(graph_id, stream_id, &bindings);
                stream_id.executes(|| graph.replay(&mut self.runner, bindings));
                Ok(())
            }
            Task::RegisterTensor(stream_id, id, data) => {
                stream_id.executes(|| self.runner.register_tensor_data_id(id, data));
                Ok(())
            }
            Task::RegisterAlias {
                stream_id,
                new_id,
                src_id,
            } => {
                stream_id.executes(|| self.runner.register_alias(new_id, src_id));
                Ok(())
            }
            Task::RegisterTensorRemote(stream_id, remote, new_id) => {
                log::trace!(
                    "Registering remote tensor (transfer {:?} from {:?})",
                    remote.capability,
                    remote.peer,
                );
                let peer = Some(format!("{:?}", remote.peer));
                self.emit_transfer(peer.clone(), TransferScope::Remote, TransferPhase::Started);
                let data = match self
                    .transfer
                    .download_tensor(remote.peer.clone(), remote.capability)
                    .await
                {
                    Some(data) => {
                        self.emit_transfer(peer, TransferScope::Remote, TransferPhase::Completed);
                        data
                    }
                    None => {
                        self.emit_transfer(peer, TransferScope::Remote, TransferPhase::Failed);
                        return Err(format!(
                            "Failed to download tensor for transfer {:?} from {:?}",
                            remote.capability, remote.peer,
                        ));
                    }
                };
                // Register on the client stream that will consume `new_id`, carried over the
                // wire — not the arbitrary tokio worker running this task.
                stream_id.executes(|| self.runner.register_tensor_data_id(new_id, data));
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
                let kind = stream_id.executes(|| self.runner.get_tensor(&tensor));
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
                stream_id.executes(|| self.runner.register_tensor_to_device(new_id, kind));
                self.emit_transfer(None, TransferScope::Local, TransferPhase::Completed);
                Ok(())
            }
            Task::ExposeTensorRemote {
                stream_id,
                tensor,
                count,
                capability,
                target,
            } => {
                log::trace!("Exposing tensor (transfer {capability:?})");
                // Same shape as `ReadTensor`: the sync part of `read_tensor_async` runs in order
                // to preserve stream ordering, but the readback + expose are detached so a
                // cross-server hand-off doesn't stall this session's op registration on a
                // GPU→host copy. A target that downloads before the expose lands simply blocks on
                // the data service's `new_tensor_notify`, so there is no race.
                let fut = stream_id.executes(|| self.runner.read_tensor_async(tensor));
                let transfer = self.transfer.clone();
                let probe = self.probe.clone();
                let session_id = self.session_id;
                let peer = Some(format!("{target:?}"));
                probe.emit(|| TelemetryEvent::Transfer {
                    session: session_id,
                    peer: peer.clone(),
                    scope: TransferScope::Remote,
                    phase: TransferPhase::Started,
                });
                spawn_detached(async move {
                    match fut.await {
                        Ok(data) => {
                            transfer.expose_data(data, count, capability, target).await;
                            probe.emit(|| TelemetryEvent::Transfer {
                                session: session_id,
                                peer,
                                scope: TransferScope::Remote,
                                phase: TransferPhase::Completed,
                            });
                        }
                        Err(e) => {
                            let reason = format!(
                                "read_tensor_async for transfer {capability:?} failed: {e:?}"
                            );
                            log::error!(
                                "read_tensor_async for transfer {capability:?} failed: {e:?}"
                            );
                            transfer.fail(capability, target, reason).await;
                            probe.emit(|| TelemetryEvent::Transfer {
                                session: session_id,
                                peer,
                                scope: TransferScope::Remote,
                                phase: TransferPhase::Failed,
                            });
                        }
                    }
                });
                Ok(())
            }
            Task::Seed(seed) => {
                self.runner.seed(seed);
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
                self.probe.emit(|| TelemetryEvent::Read {
                    session: self.session_id,
                    request: request_id,
                });
                let fut = stream_id.executes(|| self.runner.read_tensor_async(tensor));
                let sender = self.response_sender.clone();
                self.response_tasks.spawn(async move {
                    // Under an async runtime the backend reads eagerly (see `RuntimeKind::Async`),
                    // so `data` is already host-resident here — the fetch handler only serializes
                    // resident bytes and no blocking device→host copy runs on the shared runtime.
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
                self.probe.emit(|| TelemetryEvent::Sync {
                    session: self.session_id,
                    request: request_id,
                });
                let res = stream_id.executes(|| self.runner.sync());
                self.send_response(request_id, TaskResponseContent::SyncBackend(res))
                    .await
            }
            Task::DTypeUsage(request_id, dtype) => {
                let res = self.runner.dtype_usage(dtype);
                self.send_response(request_id, TaskResponseContent::DTypeUsage(res))
                    .await
            }
            Task::ProfileStart(request_id, stream_id) => {
                let res = stream_id.executes(|| self.runner.profile_start());
                self.send_response(request_id, TaskResponseContent::ProfileStart(res))
                    .await
            }
            Task::ProfileAbandon(stream_id, token) => {
                stream_id.executes(|| self.runner.profile_abandon(token));
                Ok(())
            }
            Task::ProfileEnd(request_id, stream_id, token, options) => {
                // Closing the window is sync and in order, like the read
                // above; the measurement is the device's to answer, so the
                // wait for it is detached rather than stalling the worker.
                let duration = stream_id.executes(|| self.runner.profile_end(token, options));
                let sender = self.response_sender.clone();
                self.response_tasks.spawn(async move {
                    let res = match duration {
                        Ok(duration) => Ok(duration.resolve().await.map(|ticks| ticks.duration())),
                        Err(err) => Err(err),
                    };
                    if sender
                        .send(TaskResponse {
                            content: TaskResponseContent::ProfileEnd(res),
                            id: request_id,
                        })
                        .await
                        .is_err()
                    {
                        log::warn!(
                            "Response receiver dropped before the profile for request {request_id} could be sent"
                        );
                    }
                });
                Ok(())
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

    fn emit_op(&self, stream: StreamId, op: &OperationIr) {
        self.probe
            .emit(|| TelemetryEvent::unfused_op(self.session_id, stream, op));
    }

    fn emit_transfer(&self, peer: Option<String>, scope: TransferScope, phase: TransferPhase) {
        self.probe.emit(|| TelemetryEvent::Transfer {
            session: self.session_id,
            peer,
            scope,
            phase,
        });
    }

    fn emit_graph_executed(&self, graph: GraphId, stream: StreamId, bindings: &GraphBindings) {
        self.probe.emit(|| {
            TelemetryEvent::graph_executed(self.session_id, graph, stream, serialized_len(bindings))
        });
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use crate::{
        PeerAddr, PeerId,
        shared::{LocalTransferId, TransferCapability},
    };
    use burn_backend::{DType, Shape, TensorData};
    use burn_flex::Flex;
    use burn_ir::{TensorId, TensorIr, TensorStatus};
    use std::time::Duration;

    struct NoTransfer;

    impl<B: BackendIr> TensorTransfer<B> for NoTransfer {
        async fn expose_data(&self, _: TensorData, _: u32, _: TransferCapability, _: PeerId) {}

        async fn download_tensor(&self, _: PeerAddr, _: TransferCapability) -> Option<TensorData> {
            None
        }

        async fn fail(&self, _: TransferCapability, _: PeerId, _: String) {}
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_panicking_task_still_tears_its_session_down() {
        let local_comm = Arc::new(LocalCommService::<Flex>::new());
        let (response_sender, mut responses) = mpsc::channel(1);
        let tasks = SessionHandler::spawn(
            SessionId::new(),
            TensorInterpreter::new(Default::default()),
            response_sender,
            Arc::new(NoTransfer),
            local_comm.clone(),
            TelemetryProbe::disabled(),
        );

        let stream_id = StreamId::current();
        let exposed = TensorIr {
            id: TensorId::new(1),
            shape: Shape::new([2]),
            status: TensorStatus::ReadWrite,
            dtype: DType::F32,
        };
        let never_written = TensorIr {
            id: TensorId::new(2),
            ..exposed.clone()
        };
        let data = TensorData::new(vec![1.0f32, 2.0], [2]);
        let request_id: RequestId = 0;
        for task in [
            Task::RegisterTensor(stream_id, exposed.id, data),
            Task::ExposeTensorLocal {
                stream_id,
                tensor: exposed,
                transfer_id: LocalTransferId::from(0),
            },
            Task::ReadTensor(request_id, stream_id, never_written),
        ] {
            tasks.send(task).await.unwrap();
        }

        let closed = tokio::time::timeout(Duration::from_secs(10), responses.recv()).await;
        assert!(
            matches!(closed, Ok(None)),
            "the session worker never let go"
        );
        assert!(
            local_comm.is_empty().await,
            "the exposed tensor outlived its session"
        );
        // Dropped only now, so the panic alone had to end the session.
        drop(tasks);
    }

    #[tokio::test]
    async fn a_panic_aborts_the_responses_still_in_flight() {
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = SessionHandler {
            session_id: SessionId::new(),
            runner: TensorInterpreter::new(Default::default()),
            response_sender,
            response_tasks: ResponseTasks::default(),
            transfer: Arc::new(NoTransfer),
            local_comm: Arc::new(LocalCommService::<Flex>::new()),
            graphs: Mutex::new(HashMap::new()),
            probe: TelemetryProbe::disabled(),
        };
        let stalled = handler.response_sender.clone();
        handler.response_tasks.spawn(async move {
            let _held = stalled;
            std::future::pending::<()>().await
        });

        handler.abort_responses_after_panic(Box::new("a task panicked"));
        handler.close().await;

        let closed = tokio::time::timeout(Duration::from_secs(10), responses.recv()).await;
        assert!(
            matches!(closed, Ok(None)),
            "a stalled response kept the session open"
        );
    }
}
