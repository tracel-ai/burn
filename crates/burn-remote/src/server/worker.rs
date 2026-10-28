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
//! the backend's thread-local stream (via [`StreamId::executes`]), not to reorder work. That stream
//! is the session's own server stream for the client's (see [`ServerStreams`]), never the
//! client's id itself.
//!
//! ## Why a dedicated OS thread, not a tokio task
//!
//! Some tasks are **synchronously blocking** — a same-host transfer's
//! [`RegisterTensorLocal`](crate::shared::Task::RegisterTensorLocal) waits on the rendezvous
//! (`local_comm.take`) for its source session to expose the primitive, and an
//! [`AllReduceLocal`](crate::shared::Task::AllReduceLocal) parks until *every* member arrives.
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

use burn_backend::tensor::FloatTensor;
use burn_ir::{BackendIr, GraphBindings, GraphId, HandleKind, TensorError};
use burn_router::{Graph, TensorInterpreter};
use burn_std::{ExecutionError, id::StreamId};
#[cfg(not(target_family = "wasm"))]
use std::{
    any::Any,
    panic::{self, AssertUnwindSafe},
};
#[cfg(not(target_family = "wasm"))]
use tokio::runtime::Handle;
use tokio::sync::mpsc;

use crate::server::capture::{CaptureClaim, SessionCaptures};
use crate::server::local_comm::LocalCommService;
use crate::server::spawn::{CatchPanic, ResponseTasks, spawn_detached};
use crate::server::streams::ServerStreams;
use crate::server::transfer::TensorTransfer;
use crate::shared::{
    LocalAllReducePart, RequestId, SessionId, Task, TaskResponse, TaskResponseContent,
};
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
    streams: ServerStreams,
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
    captures: SessionCaptures<B>,
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
        capture: CaptureClaim,
        probe: TelemetryProbe,
    ) -> mpsc::Sender<Task> {
        let handler = SessionHandler {
            session_id,
            runner,
            streams: ServerStreams::default(),
            response_sender,
            response_tasks: ResponseTasks::default(),
            transfer,
            local_comm,
            graphs: Mutex::new(HashMap::new()),
            captures: SessionCaptures::new(session_id, capture),
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
        self.local_comm.abandon_session(session_id).await;

        log::debug!("Session {session_id} worker draining and exiting");

        let SessionHandler {
            runner,
            response_sender,
            captures,
            ..
        } = self;
        // Before the syncs below, which a stream left recording would refuse, and before the
        // response sender, so a client that sees its session end can capture on the device again.
        captures.close(&runner);
        // Released next, so the client sees its session end even if a sync below hangs.
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
    /// Sync work runs [on the server stream](Self::on_stream) of the client stream the task names,
    /// so work the client ordered on one stream stays ordered on one stream here.
    /// Response-producing tasks carry their own [`RequestId`] for routing the response back to the
    /// right pending callback on the client. Async work (data-service transfers,
    /// `read_tensor_async`) runs without a stream context — the relevant stream id is captured into
    /// the future at construction time via `executes`.
    async fn process_task(&mut self, task: Task) -> Result<(), String> {
        match task {
            Task::RegisterOperation(stream_id, op) => {
                // An op received individually (not as part of a cached graph) is an unfused op.
                self.emit_op(stream_id, &op);
                self.on_stream(stream_id, |runner| runner.register_op(op));
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
                let graph = Graph::new(relative_graph);
                self.graphs.lock().unwrap().insert(graph_id, graph.clone());
                self.on_stream(stream_id, |runner| graph.replay(runner, bindings));
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
                self.on_stream(stream_id, |runner| graph.replay(runner, bindings));
                Ok(())
            }
            Task::RegisterTensor(stream_id, id, data) => {
                self.on_stream(stream_id, |runner| runner.register_tensor_data_id(id, data));
                Ok(())
            }
            Task::RegisterAlias {
                stream_id,
                new_id,
                src_id,
            } => {
                self.on_stream(stream_id, |runner| runner.register_alias(new_id, src_id));
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
                self.on_stream(stream_id, |runner| {
                    runner.register_tensor_data_id(new_id, data)
                });
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
                let kind = self.on_stream(stream_id, |runner| runner.get_tensor(&tensor));
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
                self.on_stream(stream_id, |runner| {
                    runner.register_tensor_to_device(new_id, kind)
                });
                self.emit_transfer(None, TransferScope::Local, TransferPhase::Completed);
                Ok(())
            }
            Task::AllReduceLocal { stream_id, part } => {
                self.all_reduce_local(stream_id, part).await
            }
            Task::ExposeTensorRemote {
                stream_id,
                tensor,
                count,
                capability,
                target,
            } => {
                log::trace!("Exposing tensor (transfer {capability:?})");
                // The sync part of `read_tensor_async` runs in order to preserve stream ordering, but the readback + expose are detached so a
                // cross-server hand-off doesn't stall this session's op registration on a
                // GPU→host copy. A target that downloads before the expose lands simply blocks on
                // the data service's `new_tensor_notify`, so there is no race.
                let fut = self.on_stream(stream_id, |runner| runner.read_tensor_async(tensor));
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
            Task::ReadTensors(request_id, stream_id, tensors) => {
                // The tensors are taken here, in stream order; only the host readback waits.
                self.probe.emit(|| TelemetryEvent::Read {
                    session: self.session_id,
                    request: request_id,
                });
                let read = self.on_stream(stream_id, |runner| runner.read_tensors_async(&tensors));
                match self.take_replay_failure(stream_id) {
                    // The tensors are still taken and read, but the client learns its replay
                    // failed rather than reading what the step before it wrote.
                    Some(failure) => self.answer_when_ready(
                        request_id,
                        async move {
                            let _ = read.await;
                            Err(failure)
                        },
                        TaskResponseContent::ReadTensors,
                    ),
                    None => {
                        self.answer_when_ready(request_id, read, TaskResponseContent::ReadTensors)
                    }
                }
                Ok(())
            }
            Task::SyncBackend(request_id, stream_id) => {
                self.probe.emit(|| TelemetryEvent::Sync {
                    session: self.session_id,
                    request: request_id,
                });
                let synced = self.on_stream(stream_id, |runner| runner.sync());
                let res = self.take_replay_failure(stream_id).map_or(synced, Err);
                self.send_response(request_id, TaskResponseContent::SyncBackend(res))
                    .await
            }
            Task::DTypeUsage(request_id, dtype) => {
                let res = self.runner.dtype_usage(dtype);
                self.send_response(request_id, TaskResponseContent::DTypeUsage(res))
                    .await
            }
            Task::ProfileStart(request_id, stream_id) => {
                let res = self.on_stream(stream_id, |runner| runner.profile_start());
                self.send_response(request_id, TaskResponseContent::ProfileStart(res))
                    .await
            }
            Task::ProfileAbandon(stream_id, token) => {
                self.on_stream(stream_id, |runner| runner.profile_abandon(token));
                Ok(())
            }
            Task::ProfileEnd(request_id, stream_id, token, options) => {
                // Closing the window is sync and in order, like the read
                // above; the measurement is the device's to answer, so the
                // wait for it is detached rather than stalling the worker.
                let duration =
                    self.on_stream(stream_id, |runner| runner.profile_end(token, options));
                let measured = async move {
                    match duration {
                        Ok(duration) => Ok(duration.resolve().await.map(|ticks| ticks.duration())),
                        Err(err) => Err(err),
                    }
                };
                self.answer_when_ready(request_id, measured, TaskResponseContent::ProfileEnd);
                Ok(())
            }
            Task::GraphPrepare(request_id, stream_id) => {
                let stream = self.streams.of(stream_id);
                let res = self.captures.prepare(&self.runner, stream);
                self.send_response(request_id, TaskResponseContent::Graph(res))
                    .await
            }
            Task::GraphStartCapture(request_id, stream_id) => {
                let stream = self.streams.of(stream_id);
                let res = self.captures.start(&self.runner, stream);
                self.send_response(request_id, TaskResponseContent::Graph(res))
                    .await
            }
            Task::GraphStopCapture(request_id, stream_id, graph) => {
                let stream = self.streams.of(stream_id);
                let res = self.captures.stop(&self.runner, stream, graph);
                self.send_response(request_id, TaskResponseContent::Graph(res))
                    .await
            }
            Task::GraphReplay(stream_id, graph) => {
                let stream = self.streams.of(stream_id);
                self.captures.replay(&self.runner, stream, graph);
                Ok(())
            }
            Task::GraphRelease(graph) => {
                self.captures.release(graph);
                Ok(())
            }
        }
    }

    /// Hands this session's member to its all-reduce and registers its result, or the error that
    /// stopped it, as `part.out`. Blocks this session's worker until every member has arrived, so
    /// the members' sessions must not wait on each other through earlier tasks.
    async fn all_reduce_local(
        &mut self,
        stream_id: StreamId,
        part: LocalAllReducePart,
    ) -> Result<(), String> {
        self.probe
            .emit(|| part.op_event(self.session_id, stream_id));
        let reduced = match self.all_reduce_input(stream_id, &part) {
            Ok(tensor) => {
                let stream = self.streams.of(stream_id);
                self.local_comm.all_reduce(&part, tensor, stream).await
            }
            Err(reason) => {
                // The other members wait for this one, so they are told it cannot come.
                self.local_comm.fail(&part, &reason).await;
                Err(reason)
            }
        };
        match reduced {
            Ok(reduced) => {
                self.on_stream(stream_id, |runner| {
                    runner.register_tensor_to_device(part.out, HandleKind::Float(reduced))
                });
                Ok(())
            }
            Err(reason) => {
                let error = TensorError::new(ExecutionError::generic(reason.clone()));
                self.runner.set_error(part.out, error);
                Err(reason)
            }
        }
    }

    /// This member's tensor, unless the op producing it failed or it is not a float.
    fn all_reduce_input(
        &mut self,
        stream_id: StreamId,
        part: &LocalAllReducePart,
    ) -> Result<FloatTensor<B>, String> {
        if let Some(reason) = self.runner.read_failure(&part.tensor.id) {
            return Err(format!("all-reduce {:?}: {reason}", part.collective));
        }
        match self.on_stream(stream_id, |runner| runner.get_tensor(&part.tensor)) {
            HandleKind::Float(tensor) => Ok(tensor),
            _ => Err(format!(
                "all-reduce {:?} takes float tensors",
                part.collective
            )),
        }
    }

    /// Run `f` on the session's server stream for the client's `stream`.
    fn on_stream<R>(
        &mut self,
        stream: StreamId,
        f: impl FnOnce(&mut TensorInterpreter<B>) -> R,
    ) -> R {
        let stream = self.streams.of(stream);
        stream.executes(|| f(&mut self.runner))
    }

    /// The replay on the client's `stream` that failed since its last read or sync.
    fn take_replay_failure(&mut self, stream: StreamId) -> Option<ExecutionError> {
        let stream = self.streams.of(stream);
        self.captures.take_replay_failure(stream)
    }

    /// Sends the answer once `ready` resolves, without holding up the ops after it. A panic is
    /// answered as an error, so the client is never left waiting.
    fn answer_when_ready<Value: Send + 'static>(
        &mut self,
        request_id: RequestId,
        ready: impl Future<Output = Result<Value, ExecutionError>> + Send + 'static,
        into_content: fn(Result<Value, ExecutionError>) -> TaskResponseContent,
    ) {
        let sender = self.response_sender.clone();
        self.response_tasks.spawn(async move {
            let answer = CatchPanic::new(ready).await.unwrap_or_else(|message| {
                Err(ExecutionError::generic(format!(
                    "The server panicked: {message}"
                )))
            });
            let response = TaskResponse {
                content: into_content(answer),
                id: request_id,
            };
            if sender.send(response).await.is_err() {
                log::warn!("Response receiver dropped before answering request {request_id}");
            }
        });
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
    use burn_ir::{CapturedGraphId, TensorId, TensorIr, TensorStatus};
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
        let local_comm = Arc::new(LocalCommService::<Flex>::new(1));
        let (response_sender, mut responses) = mpsc::channel(1);
        let tasks = SessionHandler::spawn(
            SessionId::new(),
            TensorInterpreter::new(Default::default()),
            response_sender,
            Arc::new(NoTransfer),
            local_comm.clone(),
            CaptureClaim::default(),
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
            Task::ReadTensors(request_id, stream_id, vec![never_written]),
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
        let mut handler = SessionHandler::new(response_sender, CaptureClaim::default());
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

    #[tokio::test]
    async fn a_read_that_panics_is_answered_with_an_error() {
        fn failing_read() -> Result<Vec<TensorData>, ExecutionError> {
            panic!("the backend failed")
        }
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = SessionHandler::new(response_sender, CaptureClaim::default());

        let request_id: RequestId = 7;
        handler.answer_when_ready(
            request_id,
            async { failing_read() },
            TaskResponseContent::ReadTensors,
        );

        let response = tokio::time::timeout(Duration::from_secs(10), responses.recv())
            .await
            .expect("a panicking read still answers")
            .unwrap();
        assert_eq!(response.id, request_id);
        match response.content {
            TaskResponseContent::ReadTensors(Err(error)) => {
                assert!(error.to_string().contains("the backend failed"), "{error}")
            }
            other => panic!("expected a read error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_backend_without_graphs_answers_the_capture_with_its_error() {
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = SessionHandler::new(response_sender, CaptureClaim::default());

        let request_id: RequestId = 3;
        handler
            .process_task(Task::GraphStartCapture(request_id, StreamId::current()))
            .await
            .unwrap();

        let response = responses.recv().await.unwrap();
        assert_eq!(response.id, request_id);
        match response.content {
            TaskResponseContent::Graph(Err(error)) => {
                assert!(error.to_string().contains("not supported"), "{error}")
            }
            other => panic!("expected a capture error, got {other:?}"),
        }
    }

    /// The client does not wait on a replay, so the next read of its stream answers for it, once;
    /// another stream's sync is not its to report.
    #[tokio::test]
    async fn a_failed_replay_answers_the_next_read_of_its_stream() {
        let (response_sender, mut responses) = mpsc::channel(4);
        let mut handler = SessionHandler::new(response_sender, CaptureClaim::default());
        let replayed = StreamId { value: 0 };
        let other = StreamId { value: 1 };
        let tensor = TensorIr {
            id: TensorId::new(1),
            shape: Shape::new([2]),
            status: TensorStatus::ReadWrite,
            dtype: DType::F32,
        };

        for task in [
            Task::RegisterTensor(replayed, tensor.id, TensorData::new(vec![1.0f32, 2.0], [2])),
            Task::GraphReplay(replayed, CapturedGraphId(9)),
            Task::SyncBackend(1, other),
            Task::ReadTensors(2, replayed, vec![tensor]),
        ] {
            handler.process_task(task).await.unwrap();
        }
        assert!(matches!(
            next_answer(&mut responses).await,
            TaskResponseContent::SyncBackend(Ok(()))
        ));
        match next_answer(&mut responses).await {
            TaskResponseContent::ReadTensors(Err(error)) => {
                assert!(
                    error.to_string().contains("unknown captured graph"),
                    "{error}"
                )
            }
            other => panic!("expected the replay's error, got {other:?}"),
        }
        handler
            .process_task(Task::SyncBackend(3, replayed))
            .await
            .unwrap();
        assert!(matches!(
            next_answer(&mut responses).await,
            TaskResponseContent::SyncBackend(Ok(()))
        ));
    }

    /// One capture runs per device: a second session is refused while the first holds it, and a
    /// session that ends with its capture open lets the next one capture. Flex records no graph,
    /// so the capture here never gets past its preparation.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_session_ended_mid_capture_hands_the_device_to_the_next() {
        let device = CaptureClaim::default();
        let spawn = |device: &CaptureClaim| {
            let (response_sender, responses) = mpsc::channel(4);
            let tasks = SessionHandler::spawn(
                SessionId::new(),
                TensorInterpreter::<Flex>::new(Default::default()),
                response_sender,
                Arc::new(NoTransfer),
                Arc::new(LocalCommService::<Flex>::new(1)),
                device.clone(),
                TelemetryProbe::disabled(),
            );
            (tasks, responses)
        };
        async fn prepare(
            tasks: &mpsc::Sender<Task>,
            responses: &mut mpsc::Receiver<TaskResponse>,
        ) -> Result<(), ExecutionError> {
            // Both sessions name the same client stream.
            let stream = StreamId { value: 0 };
            tasks.send(Task::GraphPrepare(0, stream)).await.unwrap();
            match responses.recv().await.unwrap().content {
                TaskResponseContent::Graph(res) => res,
                other => panic!("expected a capture answer, got {other:?}"),
            }
        }
        let (first, mut first_responses) = spawn(&device);
        let (second, mut second_responses) = spawn(&device);

        prepare(&first, &mut first_responses).await.unwrap();
        let refused = prepare(&second, &mut second_responses).await.unwrap_err();
        assert!(
            refused.to_string().contains("one capture runs"),
            "{refused}"
        );

        drop(first);
        let closed = tokio::time::timeout(Duration::from_secs(10), first_responses.recv()).await;
        assert!(matches!(closed, Ok(None)), "the first session never ended");
        prepare(&second, &mut second_responses).await.unwrap();
    }

    async fn next_answer(responses: &mut mpsc::Receiver<TaskResponse>) -> TaskResponseContent {
        tokio::time::timeout(Duration::from_secs(10), responses.recv())
            .await
            .expect("the server answers")
            .unwrap()
            .content
    }

    impl SessionHandler<Flex, NoTransfer> {
        fn new(response_sender: mpsc::Sender<TaskResponse>, capture: CaptureClaim) -> Self {
            let session_id = SessionId::new();
            Self {
                session_id,
                runner: TensorInterpreter::new(Default::default()),
                streams: ServerStreams::default(),
                response_sender,
                response_tasks: ResponseTasks::default(),
                transfer: Arc::new(NoTransfer),
                local_comm: Arc::new(LocalCommService::<Flex>::new(1)),
                graphs: Mutex::new(HashMap::new()),
                captures: SessionCaptures::new(session_id, capture),
                probe: TelemetryProbe::disabled(),
            }
        }
    }
}
