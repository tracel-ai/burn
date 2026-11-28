use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use burn_backend::ops::ReadFuture;
use burn_backend::tensor::{BoolTensor, FloatTensor};
use burn_backend::{PeakError, ProfileOptions, ProfileToken, TensorData};
use burn_ir::{
    BackendIr, GraphBindings, GraphId, HandleKind, OperationIr, TensorError, TensorId, TensorIr,
};
use burn_router::{Graph, TensorInterpreter};
use burn_std::{BoolStore, DType, ExecutionError, future::DynFut, id::StreamId};
use serde_bytes::ByteBuf;
#[cfg(not(target_family = "wasm"))]
use std::{
    any::Any,
    panic::{self, AssertUnwindSafe},
};
#[cfg(not(target_family = "wasm"))]
use tokio::runtime::Handle;
use tokio::sync::mpsc;

use super::responder::{SessionResponder, panic_as_error};
use crate::PeerId;
use crate::server::link_probe::LinkPuller;
use crate::server::local_comm::LocalCommService;
use crate::server::spawn::{CatchPanic, spawn_detached};
use crate::server::transfer::TensorTransfer;
use crate::server::{
    DeviceCaptureLock, DeviceQueries, ServerStreams, SessionCaptures, SessionExecutionModes,
};
use crate::shared::{
    LocalAllReducePart, LocalTransferId, RequestId, SessionId, Task, TaskResponse,
    TaskResponseContent, TensorRemote, TransferCapability,
};
use crate::telemetry::{
    TelemetryEvent, TelemetryProbe, TransferPhase, TransferScope, serialized_len,
};

/// A full channel makes the submit task's send wait, so it stops reading the socket rather than
/// blocking a thread or growing memory; sized for a burst of ops while the worker is mid-task.
const TASK_CHANNEL_CAPACITY: usize = 64;

/// [`SessionCaptures::prepare`] or [`SessionCaptures::start`].
type CaptureOpening<B> =
    fn(&mut SessionCaptures<B>, &TensorInterpreter<B>, StreamId) -> Result<(), ExecutionError>;

/// What a server shares with each of its sessions: the tensor transfers to other servers and
/// between its own sessions, its device queries and its telemetry.
pub struct SessionServices<B: BackendIr, T> {
    pub transfer: Arc<T>,
    pub local_comm: Arc<LocalCommService<B>>,
    pub queries: DeviceQueries<B>,
    pub probe: TelemetryProbe,
}

/// Starts the session's worker.
///
/// # Assumptions
///
/// Off the browser, runs inside the tokio runtime that owns the endpoint.
///
/// # Guarantees
///
/// The worker closes the session once every clone of the returned sender is dropped, leaving the
/// replies still pending to finish. Off the browser, it also closes the session once a task
/// panics, cancelling the replies still pending.
pub fn spawn_session_worker<B, T>(
    session_id: SessionId,
    runner: TensorInterpreter<B>,
    response_sender: mpsc::Sender<TaskResponse>,
    services: SessionServices<B, T>,
    capture_lock: DeviceCaptureLock,
) -> mpsc::Sender<Task>
where
    B: BackendIr,
    T: TensorTransfer<B>,
{
    let (sender, receiver) = mpsc::channel(TASK_CHANNEL_CAPACITY);
    SessionHandler::new(session_id, runner, response_sender, services, capture_lock)
        .drive(receiver);
    sender
}

/// Handles a session's tasks one at a time, against the session's interpreter.
///
/// # Assumptions
///
/// A task that consumes a tensor arrives after the task that produces it.
///
/// # Guarantees
///
/// A task's stream work is submitted before the next task starts. A task that fails answers with
/// its error or claims the tensors it was going to write. A failed capture replay instead answers
/// the next read or sync of its stream, which reports only the first replay that failed since the
/// last one.
pub struct SessionHandler<B, T>
where
    B: BackendIr,
    T: TensorTransfer<B>,
{
    session_id: SessionId,
    runner: TensorInterpreter<B>,
    streams: ServerStreams,
    responder: SessionResponder,
    transfer: Arc<T>,
    local_comm: Arc<LocalCommService<B>>,
    /// The graphs the client registered, by id, for its later executes to replay.
    graphs: HashMap<GraphId, Graph>,
    queries: DeviceQueries<B>,
    captures: SessionCaptures<B>,
    execution_modes: SessionExecutionModes,
    probe: TelemetryProbe,
}

impl<B, T> SessionHandler<B, T>
where
    B: BackendIr,
    T: TensorTransfer<B>,
{
    pub fn new(
        session_id: SessionId,
        runner: TensorInterpreter<B>,
        response_sender: mpsc::Sender<TaskResponse>,
        services: SessionServices<B, T>,
        capture_lock: DeviceCaptureLock,
    ) -> Self {
        let SessionServices {
            transfer,
            local_comm,
            queries,
            probe,
        } = services;
        Self {
            session_id,
            runner,
            streams: ServerStreams::default(),
            responder: SessionResponder::new(session_id, response_sender),
            transfer,
            local_comm,
            graphs: HashMap::new(),
            queries,
            captures: SessionCaptures::new(session_id, capture_lock),
            execution_modes: SessionExecutionModes::default(),
            probe,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn drive(mut self, receiver: mpsc::Receiver<Task>) {
        let handle = Handle::current();
        let session_id = self.session_id;
        std::thread::Builder::new()
            .name(format!("burn-remote-session-{session_id}"))
            .spawn(move || {
                // A panic can leave the interpreter half-applied, so the session is closed rather
                // than given another task.
                let handled = panic::catch_unwind(AssertUnwindSafe(|| {
                    handle.block_on(self.handle_tasks(receiver))
                }));
                if let Err(payload) = handled {
                    self.stop_after_panic(payload);
                }
                handle.block_on(self.close());
            })
            .expect("Failed to spawn session worker thread");
    }

    #[cfg(target_family = "wasm")]
    pub fn drive(mut self, receiver: mpsc::Receiver<Task>) {
        spawn_detached(async move {
            self.handle_tasks(receiver).await;
            self.close().await;
        });
    }

    pub async fn handle_task(&mut self, task: Task) {
        match task {
            Task::RegisterOperation(stream_id, op) => self.register_op(stream_id, op),
            Task::RegisterAndExecuteGraph {
                stream_id,
                graph_id,
                relative_graph,
                bindings,
            } => self.register_and_execute_graph(stream_id, graph_id, relative_graph, bindings),
            Task::ExecuteGraph {
                stream_id,
                graph_id,
                bindings,
            } => self.execute_graph(stream_id, graph_id, bindings),
            Task::RegisterTensor(stream_id, id, data) => {
                self.on_stream(stream_id, |runner| runner.register_tensor_data_id(id, data))
            }
            Task::RegisterAlias {
                stream_id,
                new_id,
                src_id,
            } => self.on_stream(stream_id, |runner| runner.register_alias(new_id, src_id)),
            Task::RegisterTensorRemote {
                stream_id,
                tensor,
                new_id,
                dtype,
            } => {
                self.register_remote_tensor(stream_id, tensor, new_id, dtype)
                    .await
            }
            Task::ExposeTensorLocal {
                stream_id,
                tensor,
                transfer_id,
            } => {
                self.expose_local_tensor(stream_id, tensor, transfer_id)
                    .await
            }
            Task::RegisterTensorLocal {
                stream_id,
                transfer_id,
                new_id,
                dtype,
            } => {
                self.register_local_tensor(stream_id, transfer_id, new_id, dtype)
                    .await
            }
            Task::RegisterFailure {
                stream_id,
                id,
                reason,
            } => self.on_stream(stream_id, |runner| runner.set_error(id, failure(reason))),
            Task::AllReduceLocal { stream_id, part } => {
                self.all_reduce_local(stream_id, part).await
            }
            Task::ExposeTensorRemote {
                stream_id,
                tensor,
                count,
                capability,
                target,
            } => self.expose_remote_tensor(stream_id, tensor, count, capability, target),
            Task::Seed(seed) => self.runner.seed(seed),
            Task::ReadTensors(request_id, stream_id, tensors) => {
                self.read_tensors(request_id, stream_id, tensors)
            }
            Task::SyncBackend(request_id, stream_id) => {
                self.sync_backend(request_id, stream_id).await
            }
            Task::DTypeUsage(request_id, dtype) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::DTypeUsage(self.runner.dtype_usage(dtype)),
                    )
                    .await
            }
            Task::ProfileStart(request_id, stream_id) => {
                self.respond_on_stream(
                    request_id,
                    stream_id,
                    |runner| runner.profile_start(),
                    TaskResponseContent::ProfileStart,
                )
                .await
            }
            Task::ProfileAbandon(stream_id, token) => {
                self.on_stream(stream_id, |runner| runner.profile_abandon(token))
            }
            Task::ProfileEnd(request_id, stream_id, token, options) => {
                self.end_profile(request_id, stream_id, token, options)
            }
            Task::CapturePrepare(request_id, stream_id) => {
                self.open_capture(request_id, stream_id, SessionCaptures::prepare)
                    .await
            }
            Task::CaptureEndRun(request_id, stream_id) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::CaptureEndRun(
                            self.captures
                                .end_run(&self.runner, self.streams.of(stream_id)),
                        ),
                    )
                    .await
            }
            Task::CaptureStartRun(request_id, stream_id) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::Capture(
                            self.captures
                                .start_run(&self.runner, self.streams.of(stream_id)),
                        ),
                    )
                    .await
            }
            Task::CaptureStart(request_id, stream_id) => {
                self.open_capture(request_id, stream_id, SessionCaptures::start)
                    .await
            }
            Task::CaptureStop(request_id, stream_id, graph) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::Capture(self.captures.stop(
                            &self.runner,
                            self.streams.of(stream_id),
                            graph,
                        )),
                    )
                    .await
            }
            Task::CaptureReplay(stream_id, graph) => {
                self.captures
                    .replay(&self.runner, self.streams.of(stream_id), graph)
            }
            Task::CaptureRelease(graph) => self.captures.release(graph),
            Task::DeviceQuery(request_id, id) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::DeviceQuery(
                            self.queries
                                .answer(&id, &self.runner.device())
                                .map(ByteBuf::from),
                        ),
                    )
                    .await
            }
            Task::ExecutionModeOpen(request_id, stream_id, mode) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::ExecutionModeOpen(self.execution_modes.open(
                            &self.runner,
                            self.streams.of(stream_id),
                            mode,
                        )),
                    )
                    .await
            }
            Task::ExecutionModeStatistics(request_id, token) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::ExecutionStatistics(
                            self.execution_modes.statistics(&self.runner, token),
                        ),
                    )
                    .await
            }
            Task::ExecutionModeClose(request_id, token) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::ExecutionStatistics(
                            self.execution_modes.close(&self.runner, token),
                        ),
                    )
                    .await
            }
            Task::EnvironmentCreate(request_id, name, origin) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::Environment(
                            self.runner.environment_create(name, origin),
                        ),
                    )
                    .await
            }
            Task::EnvironmentActivate(request_id, stream_id, id) => {
                self.respond_on_stream(
                    request_id,
                    stream_id,
                    |runner| runner.environment_activate(id),
                    TaskResponseContent::Environment,
                )
                .await
            }
            Task::EnvironmentActive(request_id) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::Environment(self.runner.environment_active()),
                    )
                    .await
            }
            Task::EnvironmentSave(request_id, stream_id, id, options) => {
                self.respond_on_stream(
                    request_id,
                    stream_id,
                    |runner| runner.environment_save(id, options),
                    TaskResponseContent::EnvironmentSaved,
                )
                .await
            }
            Task::EnvironmentSummarize(request_id, id) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::EnvironmentSummarized(
                            self.runner.environment_summary(id),
                        ),
                    )
                    .await
            }
            Task::EnvironmentMerge(request_id, stream_id, id, origin) => {
                self.respond_on_stream(
                    request_id,
                    stream_id,
                    |runner| runner.environment_merge(id, origin),
                    TaskResponseContent::EnvironmentMerged,
                )
                .await
            }
            Task::EnvironmentPrune(request_id, stream_id, id, options) => {
                self.respond_on_stream(
                    request_id,
                    stream_id,
                    |runner| runner.environment_prune(id, options),
                    TaskResponseContent::EnvironmentPruned,
                )
                .await
            }
            Task::DeviceDescription(request_id) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::DeviceDescription(self.runner.device_description()),
                    )
                    .await
            }
            Task::MeasureTelemetry(request_id) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::Telemetry(self.runner.measure_telemetry()),
                    )
                    .await
            }
            Task::MeasureMemory(request_id, stream_id) => {
                self.respond_on_stream(
                    request_id,
                    stream_id,
                    |runner| runner.measure_memory(),
                    TaskResponseContent::Memory,
                )
                .await
            }
            Task::MeasurePeaks(request_id, keys) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::Peaks(self.runner.measure_peaks(&keys)),
                    )
                    .await
            }
            Task::GrantLink(request_id, capability) => {
                self.grant_link(request_id, capability).await
            }
            Task::MeasureLinkLocal(request_id, capability, sizes) => {
                self.measure_local_link(request_id, capability, sizes).await
            }
            Task::ExposeLinkPull(request_id, target, pull) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::LinkExposed(
                            LinkPuller::new(&*self.transfer).expose(target, &pull).await,
                        ),
                    )
                    .await
            }
            Task::MeasureLinkPull(request_id, source, pull) => {
                self.responder
                    .respond(
                        request_id,
                        TaskResponseContent::LinkPulled(
                            LinkPuller::new(&*self.transfer).time(&source, &pull).await,
                        ),
                    )
                    .await
            }
            Task::WithdrawLinkPull(capability) => self.transfer.withdraw(capability).await,
        }
    }
}

impl<B, T> SessionHandler<B, T>
where
    B: BackendIr,
    T: TensorTransfer<B>,
{
    /// Logs why the task panicked, and cancels the answers still pending.
    #[cfg(not(target_family = "wasm"))]
    fn stop_after_panic(&mut self, payload: Box<dyn Any + Send>) {
        self.responder.abort_pending();
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

    async fn handle_tasks(&mut self, mut receiver: mpsc::Receiver<Task>) {
        log::debug!("Session {} worker started", self.session_id);
        while let Some(task) = receiver.recv().await {
            self.handle_task(task).await;
        }
    }

    /// Tear the session down in an order that releases its memory.
    async fn close(self) {
        let session_id = self.session_id;

        // Reclaim the same-host transfers this session exposed that no target took, so they
        // don't strand device memory in the shared registry.
        self.local_comm.purge_session(session_id).await;
        self.local_comm.abandon_session(session_id).await;

        log::debug!("Session {session_id} worker draining and exiting");

        let SessionHandler {
            runner,
            responder,
            captures,
            execution_modes,
            ..
        } = self;
        // Before the syncs below, which a stream left recording would refuse, and before the
        // response sender, so a client that sees its session end can capture on the device again.
        captures.close(&runner);
        // So the server runs launches again, its other sessions' included, once a session that
        // left a mode open is gone.
        execution_modes.close_all(&runner);
        // Released next, so the client sees its session end even if a sync below hangs.
        drop(responder);
        let device = runner.device();

        // So the session's tensors aren't freed with device work still queued against them.
        warn_unless_synced(session_id, "runner.sync()", runner.sync());
        warn_unless_synced(session_id, "B::sync(device)", B::sync(&device));

        // The memory the runner's handles free is reclaimable only once they drop, and a
        // long-lived server would otherwise keep it pooled across session churn.
        drop(runner);
        B::memory_cleanup(&device);
    }

    fn register_op(&mut self, stream_id: StreamId, op: OperationIr) {
        // An op received outside a cached graph is an unfused op.
        self.probe
            .emit(|| TelemetryEvent::unfused_op(self.session_id, stream_id, &op));
        self.on_stream(stream_id, |runner| runner.register_op(op));
    }

    fn register_and_execute_graph(
        &mut self,
        stream_id: StreamId,
        graph_id: GraphId,
        relative_graph: Vec<OperationIr>,
        bindings: GraphBindings,
    ) {
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
        self.graphs.insert(graph_id, graph.clone());
        self.on_stream(stream_id, |runner| graph.replay(runner, bindings));
    }

    fn execute_graph(&mut self, stream_id: StreamId, graph_id: GraphId, bindings: GraphBindings) {
        let Some(graph) = self.graphs.get(&graph_id).cloned() else {
            return self.claim_unknown_graph_outputs(graph_id, bindings);
        };
        self.emit_graph_executed(graph_id, stream_id, &bindings);
        self.on_stream(stream_id, |runner| graph.replay(runner, bindings));
    }

    fn claim_unknown_graph_outputs(&mut self, graph_id: GraphId, bindings: GraphBindings) {
        let reason = format!("Execute of unknown graph {graph_id:?}");
        // The bindings do not say which tensors are outputs, but only the outputs are unwritten.
        let error = failure(&reason);
        for (_, id) in bindings.tensors {
            self.runner.claim_unwritten(id, error.clone());
        }
        self.log_failure(&reason);
    }

    async fn register_remote_tensor(
        &mut self,
        stream: StreamId,
        remote: TensorRemote,
        new_id: TensorId,
        dtype: DType,
    ) {
        log::trace!(
            "Registering remote tensor (transfer {:?} from {:?})",
            remote.capability,
            remote.peer,
        );
        let peer = Some(&remote.peer as &dyn Debug);
        self.emit_transfer(peer, TransferScope::Remote, TransferPhase::Started);
        let downloaded = self
            .transfer
            .download_tensor(remote.peer.clone(), remote.capability)
            .await;
        let Some(data) = downloaded else {
            self.emit_transfer(peer, TransferScope::Remote, TransferPhase::Failed);
            return self.fail_tensor(
                new_id,
                format!(
                    "Failed to download tensor for transfer {:?} from {:?}",
                    remote.capability, remote.peer,
                ),
            );
        };
        self.emit_transfer(peer, TransferScope::Remote, TransferPhase::Completed);
        // On the client stream that will consume `new_id`, carried over the wire.
        self.on_stream(stream, |runner| register_data(runner, new_id, data, dtype));
    }

    /// The source side of a same-host transfer, which hands the target `tensor`'s primitive, or
    /// the failure that claimed it: the op that produced it ran before this task.
    async fn expose_local_tensor(
        &mut self,
        stream_id: StreamId,
        tensor: TensorIr,
        transfer_id: LocalTransferId,
    ) {
        let exposed = match self.runner.take_error(&tensor) {
            Some(error) => Err(error),
            None => Ok(self.on_stream(stream_id, |runner| runner.get_tensor(&tensor))),
        };
        self.local_comm
            .expose(self.session_id, transfer_id, exposed)
            .await;
    }

    /// The target side of a same-host transfer, awaited in order so the tasks after it that
    /// consume `new_id` find it. The wait blocks only this session's worker.
    async fn register_local_tensor(
        &mut self,
        stream: StreamId,
        transfer_id: LocalTransferId,
        id: TensorId,
        dtype: DType,
    ) {
        let taken = self.local_comm.take(transfer_id).await;
        let phase = finished_phase(&taken);
        match taken {
            Ok(tensor) => {
                self.on_stream(stream, |runner| register_moved(runner, id, tensor, dtype))
            }
            Err(error) => self.runner.set_error(id, error.propagated()),
        }
        self.emit_transfer(None, TransferScope::Local, phase);
    }

    /// Hands this session's member to its all-reduce and registers its result, or the error that
    /// stopped it, as `part.out`. Blocks this session's worker until every member has arrived, so
    /// the members' sessions must not wait on each other through earlier tasks.
    async fn all_reduce_local(&mut self, stream: StreamId, part: LocalAllReducePart) {
        self.probe.emit(|| part.op_event(self.session_id, stream));
        let reduced = match self.reduce_member(stream, &part).await {
            Ok(reduced) => HandleKind::Float(reduced),
            Err(reason) => return self.fail_tensor(part.out, reason),
        };
        self.on_stream(stream, |runner| {
            runner.register_tensor_to_device(part.out, reduced)
        });
    }

    async fn reduce_member(
        &mut self,
        stream_id: StreamId,
        part: &LocalAllReducePart,
    ) -> Result<FloatTensor<B>, String> {
        let input = self.all_reduce_input(stream_id, part);
        if let Err(reason) = &input {
            // The other members wait for this one, so they are told it cannot come.
            self.local_comm.fail(part, reason).await;
        }
        self.local_comm
            .all_reduce(part, input?, self.streams.of(stream_id))
            .await
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

    /// Reads `tensor` back and exposes it to `target`, detached so the tasks after it don't wait
    /// on the copy. A target that downloads first waits for the expose.
    fn expose_remote_tensor(
        &mut self,
        stream_id: StreamId,
        tensor: TensorIr,
        count: u32,
        capability: TransferCapability,
        target: PeerId,
    ) {
        log::trace!("Exposing tensor (transfer {capability:?})");
        let read = self.read_on_stream(stream_id, |runner| runner.read_tensor_async(tensor));
        let read = self.responder.track_read(read);
        self.emit_transfer(Some(&target), TransferScope::Remote, TransferPhase::Started);
        let peer = target.clone();
        let transfer = self.transfer.clone();
        let probe = self.probe.clone();
        let session = self.session_id;
        spawn_detached(async move {
            let read = read.await;
            let phase = finished_phase(&read);
            match read {
                Ok(data) => transfer.expose_data(data, count, capability, target).await,
                Err(error) => {
                    transfer
                        .fail(capability, target, log_failed_read(capability, error))
                        .await
                }
            }
            probe.emit(|| transfer_event(session, Some(&peer), TransferScope::Remote, phase));
        });
    }

    /// Takes `tensors` on the client's `stream`, in stream order, and answers `request_id` with
    /// their data once the host readback completes, without the worker waiting on it.
    ///
    /// A replay on `stream` that failed since the last read or sync is the answer instead: the
    /// tensors are still taken and read, but the client learns its replay failed rather than
    /// reading what the step before it wrote.
    fn read_tensors(&mut self, request_id: RequestId, stream: StreamId, tensors: Vec<TensorIr>) {
        self.probe.emit(|| TelemetryEvent::Read {
            session: self.session_id,
            request: request_id,
        });
        let read = self.read_on_stream(stream, |runner| runner.read_tensors_async(&tensors));
        let failure = self.take_replay_failure(stream);
        self.responder.respond_after_read(
            request_id,
            async move { failure.map_or(read.await, Err) },
            TaskResponseContent::ReadTensors,
        );
    }

    async fn sync_backend(&mut self, request_id: RequestId, stream_id: StreamId) {
        self.probe.emit(|| TelemetryEvent::Sync {
            session: self.session_id,
            request: request_id,
        });
        let synced = self.on_stream(stream_id, |runner| runner.sync());
        let synced = self.take_replay_failure(stream_id).map_or(synced, Err);
        self.responder
            .respond(request_id, TaskResponseContent::SyncBackend(synced))
            .await
    }

    /// Closes the profile window in order; the measurement is the device's to answer, so its wait
    /// is detached rather than stalling the worker.
    fn end_profile(
        &mut self,
        request_id: RequestId,
        stream_id: StreamId,
        token: ProfileToken,
        options: ProfileOptions,
    ) {
        let duration = self.on_stream(stream_id, |runner| runner.profile_end(token, options));
        let measured = async move {
            match duration {
                Ok(duration) => Ok(duration.resolve().await.map(|ticks| ticks.duration())),
                Err(err) => Err(err),
            }
        };
        self.responder
            .respond_when_ready(request_id, measured, TaskResponseContent::ProfileEnd);
    }

    /// Answers with the capture `open` starts, once no read is still copying: one on the stream
    /// would be recorded with the capture.
    async fn open_capture(
        &mut self,
        request_id: RequestId,
        stream_id: StreamId,
        open: CaptureOpening<B>,
    ) {
        self.responder.settle_reads().await;
        let opened = open(&mut self.captures, &self.runner, self.streams.of(stream_id));
        self.responder
            .respond(request_id, TaskResponseContent::Capture(opened))
            .await
    }

    async fn grant_link(&self, request_id: RequestId, capability: TransferCapability) {
        self.local_comm
            .grant_link(self.session_id, capability, self.runner.device())
            .await;
        self.responder
            .respond(request_id, TaskResponseContent::LinkGranted)
            .await
    }

    async fn measure_local_link(
        &self,
        request_id: RequestId,
        capability: TransferCapability,
        sizes: Vec<u64>,
    ) {
        let link = match self.local_comm.take_link(capability).await {
            Some(to) => self.runner.measure_link(&to, &sizes),
            None => vec![Err(PeakError::Unsupported); sizes.len()],
        };
        self.responder
            .respond(request_id, TaskResponseContent::Link(link))
            .await
    }

    /// Answers `request_id` with what `run` returns on the server stream for the client's
    /// `stream`.
    async fn respond_on_stream<R>(
        &mut self,
        request_id: RequestId,
        stream: StreamId,
        run: impl FnOnce(&mut TensorInterpreter<B>) -> R,
        into_content: fn(R) -> TaskResponseContent,
    ) {
        let answer = self.on_stream(stream, run);
        self.responder
            .respond(request_id, into_content(answer))
            .await
    }

    /// Claims `id`, which the task was going to write, with `reason`.
    fn fail_tensor(&mut self, id: TensorId, reason: String) {
        self.runner.set_error(id, failure(&reason));
        self.log_failure(&reason);
    }

    fn log_failure(&self, reason: &str) {
        log::error!("Task on session {} failed: {reason}", self.session_id);
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

    /// [`on_stream`](Self::on_stream) for a read, which the backend submits as it is made, so the
    /// worker never waits for it.
    fn read_on_stream<Data: Send + 'static>(
        &mut self,
        stream: StreamId,
        f: impl FnOnce(&mut TensorInterpreter<B>) -> DynFut<Result<Data, ExecutionError>>,
    ) -> impl ReadFuture<Data> + 'static {
        let read = self.on_stream(stream, f);
        panic_as_error(CatchPanic::from_boxed(read))
    }

    /// The replay on the client's `stream` that failed since its last read or sync.
    fn take_replay_failure(&mut self, stream: StreamId) -> Option<ExecutionError> {
        let stream = self.streams.of(stream);
        self.captures.take_replay_failure(stream)
    }

    fn emit_transfer(&self, peer: Option<&dyn Debug>, scope: TransferScope, phase: TransferPhase) {
        self.probe
            .emit(|| transfer_event(self.session_id, peer, scope, phase));
    }

    fn emit_graph_executed(&self, graph: GraphId, stream: StreamId, bindings: &GraphBindings) {
        self.probe.emit(|| {
            TelemetryEvent::graph_executed(self.session_id, graph, stream, serialized_len(bindings))
        });
    }
}

fn warn_unless_synced(session_id: SessionId, sync: &str, synced: Result<(), ExecutionError>) {
    if let Err(err) = synced {
        log::warn!("{sync} at session {session_id} close failed: {err:?}");
    }
}

fn failure(reason: impl Into<String>) -> TensorError {
    TensorError::new(ExecutionError::generic(reason))
}

/// Only bool storage may change on a transfer.
fn register_data<B: BackendIr>(
    runner: &mut TensorInterpreter<B>,
    id: TensorId,
    data: TensorData,
    dtype: DType,
) {
    let data = match dtype {
        DType::Bool(_) => data.convert_dtype(dtype),
        _ => data,
    };
    runner.register_tensor_data_id(id, data);
}

fn register_moved<B: BackendIr>(
    runner: &mut TensorInterpreter<B>,
    id: TensorId,
    tensor: HandleKind<B>,
    dtype: DType,
) {
    match (tensor, dtype) {
        (HandleKind::Bool(tensor), DType::Bool(dtype)) => register_bool(runner, id, tensor, dtype),
        (tensor, _) => runner.register_tensor_to_device(id, tensor),
    }
}

/// Uses the requested bool storage without resolving the server's defaults.
fn register_bool<B: BackendIr>(
    runner: &mut TensorInterpreter<B>,
    id: TensorId,
    tensor: BoolTensor<B>,
    dtype: BoolStore,
) {
    let tensor = B::bool_to_device(tensor, &runner.device(), dtype);
    runner.handles_mut().register_bool_tensor::<B>(&id, tensor);
}

fn transfer_event(
    session: SessionId,
    peer: Option<&dyn Debug>,
    scope: TransferScope,
    phase: TransferPhase,
) -> TelemetryEvent {
    TelemetryEvent::Transfer {
        session,
        peer: peer.map(|peer| format!("{peer:?}")),
        scope,
        phase,
    }
}

fn finished_phase<V, E>(outcome: &Result<V, E>) -> TransferPhase {
    match outcome {
        Ok(_) => TransferPhase::Completed,
        Err(_) => TransferPhase::Failed,
    }
}

fn log_failed_read(capability: TransferCapability, error: ExecutionError) -> String {
    let reason = format!("read_tensor_async for transfer {capability:?} failed: {error:?}");
    log::error!("{reason}");
    reason
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use crate::PeerAddr;
    use burn_backend::{ExecutionMode, ExecutionModeToken, ProcessMode, Shape};
    use burn_flex::Flex;
    use burn_ir::{CapturedGraphId, TensorStatus};
    use std::time::Duration;

    struct NoTensorTransfer;

    impl<B: BackendIr> TensorTransfer<B> for NoTensorTransfer {
        async fn expose_data(&self, _: TensorData, _: u32, _: TransferCapability, _: PeerId) {}

        async fn download_tensor(&self, _: PeerAddr, _: TransferCapability) -> Option<TensorData> {
            None
        }

        async fn fail(&self, _: TransferCapability, _: PeerId, _: String) {}

        async fn withdraw(&self, _: TransferCapability) {}
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_panicking_task_still_tears_its_session_down() {
        let local_comm = Arc::new(LocalCommService::<Flex>::new(1));
        let (response_sender, mut responses) = mpsc::channel(1);
        let tasks = spawn_session(
            response_sender,
            local_comm.clone(),
            DeviceCaptureLock::default(),
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
        let mut handler = handler(response_sender);
        handler.responder.respond_when_ready(
            0,
            std::future::pending::<Result<(), ExecutionError>>(),
            TaskResponseContent::SyncBackend,
        );

        handler.stop_after_panic(Box::new("a task panicked"));
        handler.close().await;

        let closed = tokio::time::timeout(Duration::from_secs(10), responses.recv()).await;
        assert!(
            matches!(closed, Ok(None)),
            "a stalled response kept the session open"
        );
    }

    #[tokio::test]
    async fn a_backend_without_graphs_answers_the_capture_with_its_error() {
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = handler(response_sender);

        let request_id: RequestId = 3;
        handler
            .handle_task(Task::CaptureStart(request_id, StreamId::current()))
            .await;

        let response = responses.recv().await.unwrap();
        assert_eq!(response.id, request_id);
        match response.content {
            TaskResponseContent::Capture(Err(error)) => {
                assert!(error.to_string().contains("not supported"), "{error}")
            }
            other => panic!("expected a capture error, got {other:?}"),
        }
    }

    /// The client does not wait on a replay, so the next read of its stream answers for it, once,
    /// with the first replay that failed: a later one ran on what the first left. Another stream's
    /// sync is not its to report.
    #[tokio::test]
    async fn a_failed_replay_answers_the_next_read_of_its_stream() {
        let (response_sender, mut responses) = mpsc::channel(4);
        let mut handler = handler(response_sender);
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
            Task::CaptureReplay(replayed, CapturedGraphId(9)),
            Task::CaptureReplay(replayed, CapturedGraphId(10)),
            Task::SyncBackend(1, other),
            Task::ReadTensors(2, replayed, vec![tensor]),
        ] {
            handler.handle_task(task).await;
        }
        assert!(matches!(
            next_answer(&mut responses).await,
            TaskResponseContent::SyncBackend(Ok(()))
        ));
        match next_answer(&mut responses).await {
            TaskResponseContent::ReadTensors(Err(error)) => {
                assert!(error.to_string().contains("(9)"), "{error}")
            }
            other => panic!("expected the replay's error, got {other:?}"),
        }
        handler.handle_task(Task::SyncBackend(3, replayed)).await;
        assert!(matches!(
            next_answer(&mut responses).await,
            TaskResponseContent::SyncBackend(Ok(()))
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_session_ended_mid_capture_hands_the_device_to_the_next() {
        let capture_lock = DeviceCaptureLock::default();
        let (first_sender, mut first_responses) = mpsc::channel(4);
        let (second_sender, mut second_responses) = mpsc::channel(4);
        let first = spawn_session(
            first_sender,
            Arc::new(LocalCommService::new(1)),
            capture_lock.clone(),
        );
        let second = spawn_session(
            second_sender,
            Arc::new(LocalCommService::new(1)),
            capture_lock,
        );
        prepare_capture(&first, &mut first_responses).await.unwrap();
        let refused = prepare_capture(&second, &mut second_responses)
            .await
            .unwrap_err();
        assert!(
            refused.to_string().contains("one capture runs"),
            "{refused}"
        );

        drop(first);
        let closed = tokio::time::timeout(Duration::from_secs(10), first_responses.recv()).await;
        assert!(matches!(closed, Ok(None)), "the first session never ended");
        prepare_capture(&second, &mut second_responses)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_local_transfer_of_a_failed_tensor_claims_the_target() {
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = handler(response_sender);
        let stream_id = StreamId::current();
        let (source, target) = (float(TensorId::new(1)), float(TensorId::new(2)));
        handler.runner.set_error(
            source.id,
            TensorError::new(ExecutionError::with_context("the source failed")),
        );

        for task in [
            Task::ExposeTensorLocal {
                stream_id,
                tensor: source,
                transfer_id: LocalTransferId::from(0),
            },
            Task::RegisterTensorLocal {
                stream_id,
                transfer_id: LocalTransferId::from(0),
                new_id: target.id,
                dtype: target.dtype,
            },
        ] {
            handler.handle_task(task).await;
        }

        let error = read_error(&mut handler, &mut responses, target).await;
        assert!(error.to_string().contains("the source failed"), "{error}");
    }

    #[tokio::test]
    async fn an_unknown_graph_claims_its_outputs() {
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = handler(response_sender);
        let stream_id = StreamId::current();
        let (input, output) = (float(TensorId::new(1)), float(TensorId::new(2)));
        let data = TensorData::new(vec![1.0f32, 2.0], [2]);

        handler
            .handle_task(Task::RegisterTensor(stream_id, input.id, data.clone()))
            .await;
        handler
            .handle_task(Task::ExecuteGraph {
                stream_id,
                graph_id: GraphId(7),
                bindings: GraphBindings {
                    tensors: vec![(TensorId::new(0), input.id), (TensorId::new(1), output.id)],
                    shapes: Vec::new(),
                    scalars: Vec::new(),
                    ranges: Vec::new(),
                },
            })
            .await;

        let error = read_error(&mut handler, &mut responses, output).await;
        assert!(error.to_string().contains("unknown graph"), "{error}");
        handler
            .handle_task(Task::ReadTensors(0, stream_id, vec![input]))
            .await;
        match next_answer(&mut responses).await {
            TaskResponseContent::ReadTensors(Ok(read)) => read[0].assert_eq(&data, true),
            other => panic!("expected the input's data, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_backend_running_every_launch_opens_no_mode() {
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = handler(response_sender);

        handler
            .handle_task(Task::ExecutionModeOpen(
                1,
                StreamId { value: 0 },
                ExecutionMode::Process(ProcessMode::CompileOnly),
            ))
            .await;

        assert!(matches!(
            next_answer(&mut responses).await,
            TaskResponseContent::ExecutionModeOpen(Ok(None))
        ));
    }

    #[tokio::test]
    async fn a_session_reads_no_mode_it_did_not_open() {
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = handler(response_sender);

        handler
            .handle_task(Task::ExecutionModeStatistics(1, foreign_mode()))
            .await;

        assert!(matches!(
            next_answer(&mut responses).await,
            TaskResponseContent::ExecutionStatistics(Err(_))
        ));
    }

    #[tokio::test]
    async fn a_session_closes_no_mode_it_did_not_open() {
        let (response_sender, mut responses) = mpsc::channel(1);
        let mut handler = handler(response_sender);

        handler
            .handle_task(Task::ExecutionModeClose(1, foreign_mode()))
            .await;

        assert!(matches!(
            next_answer(&mut responses).await,
            TaskResponseContent::ExecutionStatistics(Err(_))
        ));
    }

    fn services(
        local_comm: Arc<LocalCommService<Flex>>,
    ) -> SessionServices<Flex, NoTensorTransfer> {
        SessionServices {
            transfer: Arc::new(NoTensorTransfer),
            local_comm,
            queries: DeviceQueries::default(),
            probe: TelemetryProbe::disabled(),
        }
    }

    fn handler(
        response_sender: mpsc::Sender<TaskResponse>,
    ) -> SessionHandler<Flex, NoTensorTransfer> {
        SessionHandler::new(
            SessionId::new(),
            TensorInterpreter::new(Default::default()),
            response_sender,
            services(Arc::new(LocalCommService::new(1))),
            DeviceCaptureLock::default(),
        )
    }

    fn spawn_session(
        response_sender: mpsc::Sender<TaskResponse>,
        local_comm: Arc<LocalCommService<Flex>>,
        capture_lock: DeviceCaptureLock,
    ) -> mpsc::Sender<Task> {
        spawn_session_worker(
            SessionId::new(),
            TensorInterpreter::new(Default::default()),
            response_sender,
            services(local_comm),
            capture_lock,
        )
    }

    /// Flex records no graph, so a capture here never gets past its preparation.
    async fn prepare_capture(
        tasks: &mpsc::Sender<Task>,
        responses: &mut mpsc::Receiver<TaskResponse>,
    ) -> Result<(), ExecutionError> {
        // Every session names the same client stream.
        let stream = StreamId { value: 0 };
        tasks.send(Task::CapturePrepare(0, stream)).await.unwrap();
        match next_answer(responses).await {
            TaskResponseContent::Capture(res) => res,
            other => panic!("expected a capture answer, got {other:?}"),
        }
    }

    fn foreign_mode() -> ExecutionModeToken {
        ExecutionModeToken {
            id: 1,
            opened_on: StreamId { value: 0 },
        }
    }

    async fn next_answer(responses: &mut mpsc::Receiver<TaskResponse>) -> TaskResponseContent {
        tokio::time::timeout(Duration::from_secs(10), responses.recv())
            .await
            .expect("the server answers")
            .unwrap()
            .content
    }

    fn float(id: TensorId) -> TensorIr {
        TensorIr {
            id,
            shape: Shape::new([2]),
            status: TensorStatus::ReadOnly,
            dtype: DType::F32,
        }
    }

    async fn read_error(
        handler: &mut SessionHandler<Flex, NoTensorTransfer>,
        responses: &mut mpsc::Receiver<TaskResponse>,
        tensor: TensorIr,
    ) -> ExecutionError {
        handler
            .handle_task(Task::ReadTensors(0, StreamId::current(), vec![tensor]))
            .await;
        match next_answer(responses).await {
            TaskResponseContent::ReadTensors(Err(error)) => error,
            other => panic!("expected a read error, got {other:?}"),
        }
    }
}
