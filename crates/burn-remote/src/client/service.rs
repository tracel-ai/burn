use crate::PeerAddr;
use crate::metrics::{MetricSide, TrafficMetrics};
use crate::shared::{
    LocalTransferId, PROTOCOL_VERSION, RemoteMessage, RequestId, SessionId, SessionInfo,
    SessionInit, Task, TaskResponse, TaskResponseContent, TensorRemote, TransferCapability,
};
use burn_backend::{
    DTypeUsageSet, ExecutionError, TensorData,
    backend::{DeviceId, DeviceService, ServerUtilitiesHandle},
};
#[cfg(feature = "websocket")]
use burn_communication::{CommunicationChannel, Message, ProtocolClient};
use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_std::{DType, DeviceSettings, id::StreamId};
// Only the native `sync` path captures a backtrace; the wasm path returns without blocking.
#[cfg(not(target_family = "wasm"))]
use burn_std::backtrace::BackTrace;
use std::sync::{Arc, OnceLock};
use tokio::sync::oneshot;

mod batch;
mod pending;
mod registry;
mod writer;

use batch::OutgoingBatch;
use pending::{PendingResponses, Responder};
use writer::SubmitWriter;

pub(crate) use registry::{RemoteEndpoint, endpoint_for, register_endpoint};
use registry::{device_count_cell, settings_cell};
pub(crate) use registry::{device_count_for, has_settings, new_tensor_id, settings_for};

pub(crate) enum SubmitChannel {
    #[cfg(feature = "iroh")]
    Iroh(iroh::endpoint::SendStream),
    #[cfg(feature = "websocket")]
    WebSocket(Box<burn_communication::websocket::WsClientChannel>),
}

impl SubmitChannel {
    pub(crate) async fn send(&mut self, bytes: bytes::Bytes) -> Result<(), String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => crate::node::send_frame(stream, &bytes).await,
            #[cfg(feature = "websocket")]
            Self::WebSocket(channel) => channel
                .send(Message::new(bytes))
                .await
                .map_err(|err| err.to_string()),
        }
    }

    async fn close(&mut self) -> Result<(), String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => stream
                .finish()
                .map_err(|err| format!("Failed to finish Iroh session stream: {err}")),
            #[cfg(feature = "websocket")]
            Self::WebSocket(channel) => channel.close().await.map_err(|err| err.to_string()),
        }
    }
}

enum ResponseChannel {
    #[cfg(feature = "iroh")]
    Iroh(iroh::endpoint::RecvStream),
    #[cfg(feature = "websocket")]
    WebSocket(Box<burn_communication::websocket::WsClientChannel>),
}

impl ResponseChannel {
    #[allow(unused_variables)]
    async fn send(&mut self, bytes: bytes::Bytes) -> Result<(), String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(_) => Err("Cannot send through an Iroh receive stream".into()),
            #[cfg(feature = "websocket")]
            Self::WebSocket(channel) => channel
                .send(Message::new(bytes))
                .await
                .map_err(|err| err.to_string()),
        }
    }

    async fn recv(&mut self) -> Result<Option<bytes::Bytes>, String> {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(stream) => crate::node::recv_frame(stream)
                .await
                .map(|frame| frame.map(bytes::Bytes::from)),
            #[cfg(feature = "websocket")]
            Self::WebSocket(channel) => channel
                .recv()
                .await
                .map(|message| message.map(|message| message.data))
                .map_err(|err| err.to_string()),
        }
    }

    fn requires_init(&self) -> bool {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(_) => false,
            #[cfg(feature = "websocket")]
            Self::WebSocket(_) => true,
        }
    }
}

/// All the state owned by the device-runner thread for a single remote device.
///
/// `RemoteService` lives behind a [`DeviceHandle`](burn_backend::DeviceHandle); every call
/// from the `RouterClient` shim hops onto the runner thread via the device handle's
/// `submit` / `submit_blocking`, so the service has exclusive access to the connection,
/// the callback map, and the outgoing task buffer without any locking on its own state.
///
/// The service mirrors that submit-style API internally: fire-and-forget calls go
/// through [`submit`](Self::submit) (push onto the [`batch`](Self::batch), flush once it
/// reaches the configured flush threshold), and response-producing calls go through
/// [`submit_blocking`](Self::submit_blocking) (push, then flush right away so the request
/// is enqueued before we await the oneshot).
///
/// The moving parts are split into focused types: [`OutgoingBatch`] buffers tasks and
/// decides when to flush, [`SubmitWriter`] owns the socket and serializes frames off the
/// runner thread, and [`PendingResponses`] correlates response-producing requests with the
/// caller awaiting each reply.
///
/// All tokio work — connecting, the writer task, awaiting responses, the response-demux
/// task — happens inside the runtime owned by this struct. The caller never sees a runtime
/// handle.
pub struct RemoteService {
    runtime: ServiceRuntime,
    /// Where to connect on first use. The connection is established lazily (see
    /// [`ensure_connected`](Self::ensure_connected)) rather than in [`init`](Self::init),
    /// because cubecl holds a process-global device-registry lock across `init` — opening the
    /// sockets there would serialize every remote device's setup behind that lock.
    endpoint: RemoteEndpoint,
    device_index: u32,
    /// Owns the request socket and serializes outgoing frames off the runner thread.
    /// `None` until the first task (or a settings read) triggers the lazy connect.
    writer: Option<SubmitWriter>,
    /// Buffers outgoing tasks and signals when a batch is ready for the wire.
    batch: OutgoingBatch,
    /// Request-id allocation + the callbacks awaiting response-producing tasks.
    pending: PendingResponses,
    /// Accumulates this device's op-graph caching traffic savings (logged when remote logging is
    /// enabled). Lives here rather than in a global so each device measures its own traffic, with no
    /// locking — the service has exclusive access on the runner thread.
    metrics: TrafficMetrics,
    /// Shared cell populated from the init handshake (read by `RemoteDevice::defaults`).
    settings: Arc<OnceLock<DeviceSettings>>,
    /// Shared cell populated from the init handshake (read by `RemoteDevice::enumerate`).
    device_count: Arc<OnceLock<u32>>,
    session_id: SessionId,
    closed: bool,
}

impl DeviceService for RemoteService {
    fn init(device_id: DeviceId) -> Self {
        let (id, endpoint, device_index) = Self::resolve_endpoint(device_id);
        let runtime = ServiceRuntime::for_endpoint(&endpoint);
        let session_id = SessionId::new();

        // Lazy connect: `init` must return promptly. cubecl holds a process-global
        // device-registry lock across this call (to make device-handle creation atomic), so
        // doing the blocking network connect + handshake here would serialize every remote
        // device's setup behind that lock — N devices would connect strictly one at a time.
        // Instead we record the endpoint and open the sockets on the first real use, off the
        // lock and on the device-runner thread (see `ensure_connected`).
        Self {
            runtime,
            endpoint,
            device_index,
            writer: None,
            batch: {
                let cfg = burn_std::config::config();
                let remote = cfg.remote();
                OutgoingBatch::new(remote.flush_threshold, remote.flush_bytes_threshold)
            },
            pending: PendingResponses::new(),
            metrics: TrafficMetrics::new(MetricSide::Client),
            settings: settings_cell(id),
            device_count: device_count_cell(id),
            session_id,
            closed: false,
        }
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        // The remote backend surfaces device settings through the endpoint registry
        // (`RemoteDevice::defaults` → `settings_for`), not through this handle, so the device
        // layer never reads what we return here. Handing back an empty handle keeps `init` —
        // and the cubecl lock it runs under — free of the blocking network handshake.
        Arc::new(())
    }
}

/// Build the multi-threaded tokio runtime that hosts the connection, the writer task, and
/// the response-demux task. IO is enabled for the websocket; the runner thread enters it
/// only via `block_on`.
#[cfg(feature = "websocket")]
fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .build()
        .expect("Can build tokio runtime for remote service")
}

/// Executor that runs a remote session's writer and response-demux tasks.
///
/// On native targets this is the Tokio runtime owning the connection (a shared handle for the
/// Iroh transport, an owned runtime for the legacy websocket). In the browser there is no Tokio
/// runtime: Iroh runs on the JS event loop, so tasks are spawned with
/// [`wasm_bindgen_futures::spawn_local`] and blocking calls are unavailable (the synchronous
/// connect path is replaced by [`RemoteClient::connect_async`](super::RemoteClient)).
pub(crate) enum ServiceRuntime {
    #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
    Shared(tokio::runtime::Handle),
    #[cfg(feature = "websocket")]
    Owned(tokio::runtime::Runtime),
    #[cfg(all(feature = "iroh", target_family = "wasm"))]
    WasmLocal,
}

/// Handle to a spawned session task. Joinable on native; a no-op placeholder in the browser,
/// where detached `spawn_local` tasks simply run to completion on the event loop.
pub(crate) struct SpawnHandle {
    #[cfg(not(target_family = "wasm"))]
    inner: tokio::task::JoinHandle<()>,
}

impl ServiceRuntime {
    fn for_endpoint(endpoint: &RemoteEndpoint) -> Self {
        match endpoint {
            #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
            RemoteEndpoint::Iroh { node, .. } => Self::Shared(node.runtime()),
            #[cfg(all(feature = "iroh", target_family = "wasm"))]
            RemoteEndpoint::Iroh { .. } => Self::WasmLocal,
            #[cfg(feature = "websocket")]
            RemoteEndpoint::WebSocket { .. } => Self::Owned(build_runtime()),
        }
    }

    pub(crate) fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        match self {
            #[cfg(all(feature = "iroh", not(target_family = "wasm")))]
            Self::Shared(handle) => handle.block_on(future),
            #[cfg(feature = "websocket")]
            Self::Owned(runtime) => runtime.block_on(future),
            #[cfg(all(feature = "iroh", target_family = "wasm"))]
            Self::WasmLocal => {
                core::mem::drop(future);
                panic!(
                    "Blocking remote calls are not supported on wasm. Establish the session with \
                     `RemoteDevice::connect_async(...).await` and read tensors with \
                     `into_data_async().await`."
                )
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn spawn<F>(&self, future: F) -> SpawnHandle
    where
        F: core::future::Future<Output = ()> + Send + 'static,
    {
        let inner = match self {
            #[cfg(feature = "iroh")]
            Self::Shared(handle) => handle.spawn(future),
            #[cfg(feature = "websocket")]
            Self::Owned(runtime) => runtime.spawn(future),
        };
        SpawnHandle { inner }
    }

    /// Spawn a session task on the browser event loop. The Iroh streams these tasks own are not
    /// `Send`, which is why the wasm path uses `spawn_local` rather than the native `spawn`.
    #[cfg(target_family = "wasm")]
    pub(crate) fn spawn<F>(&self, future: F) -> SpawnHandle
    where
        F: core::future::Future<Output = ()> + 'static,
    {
        wasm_bindgen_futures::spawn_local(future);
        SpawnHandle {}
    }

    /// Wait for a spawned task to finish. No-op in the browser.
    #[cfg(not(target_family = "wasm"))]
    fn join(&self, handle: SpawnHandle) {
        let _ = self.block_on(handle.inner);
    }
}

/// Construction helpers for [`RemoteService::init`], one per step of bringing a connection
/// up. Kept separate from the public submit-style API below.
impl RemoteService {
    /// Resolve a device id to its registry index, parsed network [`Address`], and the device
    /// index to select on the server.
    fn resolve_endpoint(device_id: DeviceId) -> (u32, RemoteEndpoint, u32) {
        let id = device_id.index_id as u32;
        let (endpoint, device_index) = endpoint_for(id)
            .unwrap_or_else(|| panic!("No endpoint registered for device id {device_id}"));
        (id, endpoint, device_index)
    }

    /// Open the submit and fetch channels. Done up front so a missing server surfaces here
    /// rather than on the first op, and the demux/writer tasks can be spawned on already-open
    /// streams.
    async fn open_channels(
        endpoint: &RemoteEndpoint,
    ) -> Result<(SubmitChannel, ResponseChannel), String> {
        match endpoint {
            #[cfg(feature = "iroh")]
            RemoteEndpoint::Iroh { node, peer, .. } => {
                let (send, recv) = node
                    .open_stream(
                        &PeerAddr::Iroh(peer.clone()),
                        crate::node::StreamKind::Session,
                    )
                    .await?;
                Ok((SubmitChannel::Iroh(send), ResponseChannel::Iroh(recv)))
            }
            #[cfg(feature = "websocket")]
            RemoteEndpoint::WebSocket { address, .. } => {
                use burn_communication::websocket::WsClient;
                let submit = WsClient::connect(address.clone(), "submit")
                    .await
                    .map_err(|err| connect_error("submit", &endpoint.peer_addr(), &err))?;
                let fetch = WsClient::connect(address.clone(), "fetch")
                    .await
                    .map_err(|err| connect_error("fetch", &endpoint.peer_addr(), &err))?;
                Ok((
                    SubmitChannel::WebSocket(Box::new(submit)),
                    ResponseChannel::WebSocket(Box::new(fetch)),
                ))
            }
        }
    }

    /// Native synchronous wrapper over [`open_channels`](Self::open_channels): blocks the
    /// runner thread until the streams are open.
    #[cfg(not(target_family = "wasm"))]
    fn connect_streams(
        runtime: &ServiceRuntime,
        endpoint: &RemoteEndpoint,
    ) -> (SubmitChannel, ResponseChannel) {
        runtime
            .block_on(Self::open_channels(endpoint))
            .unwrap_or_else(|err: String| panic!("{err}"))
    }

    /// Send the session-init handshake on both streams and wait for the device settings the
    /// server replies with on the response stream. Both streams carry the same `Vec<RemoteMessage>`
    /// wire format; the handshake is just a single-element batch.
    async fn handshake_async(
        request: &mut SubmitChannel,
        response: &mut ResponseChannel,
        endpoint: &RemoteEndpoint,
        session_id: SessionId,
        device_index: u32,
    ) -> (DeviceSettings, u32) {
        let init_bytes: bytes::Bytes = rmp_serde::to_vec(&vec![RemoteMessage::Init(
            SessionInit::new(session_id, device_index, endpoint.authorization().to_vec()),
        )])
        .expect("Can serialize RemoteMessage::Init")
        .into();

        let result: Result<(DeviceSettings, u32), String> = async {
            request.send(init_bytes.clone()).await?;
            if response.requires_init() {
                response.send(init_bytes).await?;
            }

            let msg = response
                .recv()
                .await?
                .expect("Server disconnected during initialization");
            let reply: TaskResponse =
                rmp_serde::from_slice(&msg).expect("Can deserialize init handshake payload");

            match reply.content {
                TaskResponseContent::Init(SessionInfo {
                    version,
                    settings,
                    device_count,
                    ..
                }) => {
                    if version != PROTOCOL_VERSION {
                        panic!(
                            "Server uses Burn Remote protocol version {version}, expected {PROTOCOL_VERSION}"
                        );
                    }
                    Ok((settings, device_count))
                }
                other => panic!("Expected Init response, got {other:?}"),
            }
        }
        .await;

        result.unwrap_or_else(|err| {
            panic!(
                "Failed to initialize remote session at {}: {err}",
                endpoint.peer_addr()
            )
        })
    }

    /// Native synchronous wrapper over [`handshake_async`](Self::handshake_async).
    #[cfg(not(target_family = "wasm"))]
    fn handshake(
        runtime: &ServiceRuntime,
        request: &mut SubmitChannel,
        response: &mut ResponseChannel,
        endpoint: &RemoteEndpoint,
        session_id: SessionId,
        device_index: u32,
    ) -> (DeviceSettings, u32) {
        runtime.block_on(Self::handshake_async(
            request,
            response,
            endpoint,
            session_id,
            device_index,
        ))
    }

    /// Spawn the response-demux task: route each [`TaskResponse`] to its pending callback by
    /// [`RequestId`] via the [`Responder`]. Lives on the service runtime; exits when the
    /// response stream closes.
    fn spawn_response_demux(
        runtime: &ServiceRuntime,
        mut response: ResponseChannel,
        responder: Responder,
    ) {
        // Detached: the task owns the response stream and runs until it closes.
        let _demux = runtime.spawn(async move {
            loop {
                match response.recv().await {
                    Ok(Some(msg)) => {
                        let reply: TaskResponse = match rmp_serde::from_slice(&msg) {
                            Ok(r) => r,
                            Err(err) => {
                                log::error!("Failed to deserialize remote response: {err:?}");
                                continue;
                            }
                        };
                        if !responder.complete(reply.id, reply.content) {
                            log::warn!("No pending callback for response id {:?}", reply.id);
                        }
                    }
                    Ok(None) => {
                        log::warn!("Remote response stream closed");
                        break;
                    }
                    Err(err) => {
                        log::warn!("Remote response stream error: {err:?}");
                        break;
                    }
                }
            }

            // The response stream is gone (clean close or error): the server will never answer
            // any in-flight or future request on this connection. Fail every waiting caller and
            // gate new ones so they error out instead of blocking forever on a dead server.
            responder.disconnect();
        });
    }
}

/// Session state captured from a [`RemoteService`] to drive an asynchronous (wasm) connect.
#[cfg(target_family = "wasm")]
pub(crate) struct WasmConnectPlan {
    endpoint: RemoteEndpoint,
    session_id: SessionId,
    device_index: u32,
    responder: Responder,
}

/// A session opened by [`wasm_connect`], ready to be installed back into the service.
#[cfg(target_family = "wasm")]
pub(crate) struct WasmConnected {
    writer: SubmitWriter,
    settings: DeviceSettings,
    device_count: u32,
}

/// Open and hand-shake a session on the browser event loop.
///
/// This is the async counterpart of [`RemoteService::ensure_connected`]: it runs the parts that
/// would block (connecting the Iroh streams, the init round-trip) with `.await`, then spawns the
/// response-demux and writer tasks with `spawn_local`. The returned [`WasmConnected`] is `Send`,
/// so the caller can install it back into the service through the device handle.
#[cfg(target_family = "wasm")]
pub(crate) async fn wasm_connect(plan: WasmConnectPlan) -> WasmConnected {
    let runtime = ServiceRuntime::WasmLocal;

    let (mut request, mut response) = RemoteService::open_channels(&plan.endpoint)
        .await
        .unwrap_or_else(|err| panic!("{err}"));
    let (settings, device_count) = RemoteService::handshake_async(
        &mut request,
        &mut response,
        &plan.endpoint,
        plan.session_id,
        plan.device_index,
    )
    .await;

    RemoteService::spawn_response_demux(&runtime, response, plan.responder);
    let writer = SubmitWriter::spawn(&runtime, request);

    WasmConnected {
        writer,
        settings,
        device_count,
    }
}

/// Actionable panic message for a failed channel connect.
#[cfg(feature = "websocket")]
fn connect_error<E: std::fmt::Debug>(route: &str, peer: &PeerAddr, err: &E) -> String {
    format!(
        "Failed to open remote '{route}' channel to {peer}: {err:?}. \
         Is a `burn-remote` compute node running at that peer?"
    )
}

impl RemoteService {
    /// Buffer a fire-and-forget op. The buffer is flushed automatically once it reaches the
    /// configured flush threshold.
    pub fn register_op(&mut self, stream_id: StreamId, op: OperationIr) {
        // An op streamed individually (not part of a cached graph) is an unfused op.
        self.metrics.record_unfused_op(&op);
        self.submit_task(Task::RegisterOperation(stream_id, op));
    }

    /// Buffer a fire-and-forget "register a reusable op-graph and run its first invocation" task.
    pub fn register_and_execute_graph(
        &mut self,
        stream_id: StreamId,
        graph_id: burn_ir::GraphId,
        relative_graph: Vec<OperationIr>,
        bindings: burn_ir::GraphBindings,
    ) {
        // The first invocation both registers (one-time graph cost) and executes (a replay).
        self.metrics.record_registration(graph_id, &relative_graph);
        self.metrics.record_execution(graph_id, &bindings);
        self.submit_task(Task::RegisterAndExecuteGraph {
            stream_id,
            graph_id,
            relative_graph,
            bindings,
        });
    }

    /// Buffer a fire-and-forget "execute a registered graph" task.
    pub fn execute_graph(
        &mut self,
        stream_id: StreamId,
        graph_id: burn_ir::GraphId,
        bindings: burn_ir::GraphBindings,
    ) {
        self.metrics.record_execution(graph_id, &bindings);
        self.submit_task(Task::ExecuteGraph {
            stream_id,
            graph_id,
            bindings,
        });
    }

    pub fn register_tensor(&mut self, stream_id: StreamId, id: TensorId, data: TensorData) {
        self.submit_task(Task::RegisterTensor(stream_id, id, data));
        self.flush();
    }

    /// Buffer a fire-and-forget "alias `src_id` under `new_id`" task. FIFO submission keeps it
    /// after whatever materialized `src_id`, so the server has the source handle when it runs.
    pub fn register_alias(&mut self, stream_id: StreamId, new_id: TensorId, src_id: TensorId) {
        self.submit_task(Task::RegisterAlias {
            stream_id,
            new_id,
            src_id,
        });
    }

    /// Register a tensor produced by a cross-server transfer, flushing immediately.
    ///
    /// This is a coordination point with no following barrier on this client — the caller
    /// (`change_backend`) switches to the target client right after submitting it — so we
    /// flush right away instead of relying on a later op to piggy-back the flush. Otherwise
    /// the task would sit buffered below the flush threshold and the target server would
    /// never start the download.
    pub fn register_tensor_remote(
        &mut self,
        stream_id: StreamId,
        tensor: TensorRemote,
        new_id: TensorId,
    ) {
        self.submit_task(Task::RegisterTensorRemote(stream_id, tensor, new_id));
        self.flush();
    }

    /// Expose a tensor for a cross-server transfer, flushing immediately.
    ///
    /// Flushed for the same reason as [`register_tensor_remote`](Self::register_tensor_remote):
    /// the source server must actually receive the expose so the target server's download can
    /// complete, and the caller has no later barrier on this client to carry the flush.
    pub fn expose_tensor_remote(
        &mut self,
        stream_id: StreamId,
        tensor: TensorIr,
        count: u32,
        capability: TransferCapability,
        target: crate::PeerId,
    ) {
        self.submit_task(Task::ExposeTensorRemote {
            stream_id,
            tensor,
            count,
            capability,
            target,
        });
        self.flush();
    }

    /// Expose a tensor for a same-host transfer, flushing immediately.
    ///
    /// Source side of the local path; flushed for the same reason as
    /// [`expose_tensor_remote`](Self::expose_tensor_remote).
    pub fn expose_tensor_local(
        &mut self,
        stream_id: StreamId,
        tensor: TensorIr,
        transfer_id: LocalTransferId,
    ) {
        self.submit_task(Task::ExposeTensorLocal {
            stream_id,
            tensor,
            transfer_id,
        });
        self.flush();
    }

    /// Register a tensor produced by a same-host transfer, flushing immediately.
    ///
    /// Target side of the local path; flushed for the same reason as
    /// [`register_tensor_remote`](Self::register_tensor_remote).
    pub fn register_tensor_local(
        &mut self,
        stream_id: StreamId,
        transfer_id: LocalTransferId,
        new_id: TensorId,
    ) {
        self.submit_task(Task::RegisterTensorLocal {
            stream_id,
            transfer_id,
            new_id,
        });
        self.flush();
    }

    pub fn seed(&mut self, seed: u64) {
        self.submit_task(Task::Seed(seed));
    }

    /// Initiate a tensor read. The returned receiver resolves when the server response
    /// arrives.
    ///
    /// The request id rides on the task itself; the server echoes it back so the
    /// response-demux task can hand the response to the right pending callback.
    pub fn read_tensor(
        &mut self,
        stream_id: StreamId,
        tensor: TensorIr,
    ) -> oneshot::Receiver<TaskResponseContent> {
        self.submit_request(|id| Task::ReadTensor(id, stream_id, tensor))
    }

    pub fn sync(&mut self, stream_id: StreamId) -> Result<(), ExecutionError> {
        #[cfg(not(target_family = "wasm"))]
        {
            let rx = self.submit_request(|id| Task::SyncBackend(id, stream_id));
            match self.runtime.block_on(rx) {
                Ok(TaskResponseContent::SyncBackend(res)) => res,
                Ok(other) => panic!("Invalid response for SyncBackend: {other:?}"),
                Err(_) => Err(ExecutionError::Generic {
                    reason: "Remote response channel closed before sync completed".into(),
                    backtrace: BackTrace::capture(),
                }),
            }
        }
        #[cfg(target_family = "wasm")]
        {
            // The browser thread cannot block to await completion. Flushing forces every pending
            // operation onto the wire; the writer's FIFO ordering means anything submitted after
            // this `sync` still executes after it on the peer, and results are observed through
            // async reads. There is no host-side barrier to wait on, so we return immediately.
            let _ = stream_id;
            self.flush();
            Ok(())
        }
    }

    pub fn dtype_usage(&mut self, dtype: DType) -> DTypeUsageSet {
        let rx = self.submit_request(|id| Task::DTypeUsage(id, dtype));
        match self.runtime.block_on(rx) {
            Ok(TaskResponseContent::DTypeUsage(set)) => set,
            Ok(other) => panic!("Invalid response for DTypeUsage: {other:?}"),
            Err(_) => panic!("Remote response channel closed before dtype_usage completed"),
        }
    }

    /// Buffer a fire-and-forget compute task. Thin wrapper over [`submit`](Self::submit)
    /// that wraps the task in [`RemoteMessage::Task`].
    fn submit_task(&mut self, task: Task) {
        self.submit(RemoteMessage::Task(task));
    }

    /// Issue a response-producing compute task: allocate its [`RequestId`], register the
    /// pending callback, and flush immediately so it's enqueued before the caller awaits
    /// the returned receiver. `make_task` builds the task from the freshly allocated id.
    fn submit_request(
        &mut self,
        make_task: impl FnOnce(RequestId) -> Task,
    ) -> oneshot::Receiver<TaskResponseContent> {
        let request_id = self.pending.next_id();
        let rx = self.pending.register(request_id);
        self.submit_blocking(RemoteMessage::Task(make_task(request_id)));
        rx
    }

    /// Append a task to the outgoing buffer; flush only once it hits the threshold.
    ///
    /// Use this for fire-and-forget tasks. The runner thread is single-threaded, so
    /// pushing into the buffer is just a `Vec::push` — no locking, no tokio hop, no
    /// network send.
    fn submit(&mut self, task: RemoteMessage) {
        if self.batch.push(task) {
            self.flush();
        }
    }

    /// Append a task and flush the buffer immediately.
    ///
    /// Use this when the caller is about to await a response: flushing enqueues the batch
    /// to the writer task, and the FIFO writer guarantees it reaches the wire before any
    /// later frame — so the response can't be missed. We no longer block on the actual
    /// send; correctness comes from FIFO enqueue order plus the oneshot await.
    fn submit_blocking(&mut self, task: RemoteMessage) {
        self.batch.push(task);
        self.flush();
    }

    /// Open the submit/fetch sockets and run the init handshake — exactly once.
    ///
    /// Called lazily from the device-runner thread: from [`flush`](Self::flush) on the first
    /// task, or via [`RemoteClient::ensure_connected`](super::RemoteClient) on the settings
    /// path. Never from [`init`](Self::init), so the blocking connect + handshake runs off
    /// cubecl's global device-registry lock and devices connect in parallel. Always runs on
    /// the single runner thread, so the idempotent check needs no locking.
    pub fn ensure_connected(&mut self) {
        if self.writer.is_some() {
            return;
        }

        #[cfg(not(target_family = "wasm"))]
        {
            log::debug!(
                "Connecting to {} (device {}) ...",
                self.endpoint.peer_addr(),
                self.device_index
            );
            let (mut request, mut response) = Self::connect_streams(&self.runtime, &self.endpoint);
            let (settings, device_count) = Self::handshake(
                &self.runtime,
                &mut request,
                &mut response,
                &self.endpoint,
                self.session_id,
                self.device_index,
            );

            // Publish to the shared cells so `RemoteDevice::defaults`/`enumerate` can read them.
            let _ = self.settings.set(settings);
            let _ = self.device_count.set(device_count);

            Self::spawn_response_demux(&self.runtime, response, self.pending.responder());
            self.writer = Some(SubmitWriter::spawn(&self.runtime, request));
        }

        #[cfg(target_family = "wasm")]
        panic!(
            "Remote session to {} is not connected. On wasm, establish it with \
             `RemoteDevice::connect_async(...).await` before running tensor operations.",
            self.endpoint.peer_addr()
        );
    }

    /// Capture everything needed to open the session asynchronously, or `None` if it is already
    /// connected. The browser cannot block the runner thread, so the connect + handshake runs
    /// off the device handle (see [`wasm_connect`]) and the result is handed back through
    /// [`wasm_install`](Self::wasm_install). Every captured value is `Send`, so it can cross the
    /// device handle even though the Iroh streams it later owns are not.
    #[cfg(target_family = "wasm")]
    pub(crate) fn wasm_connect_plan(&mut self) -> Option<WasmConnectPlan> {
        if self.writer.is_some() {
            return None;
        }
        Some(WasmConnectPlan {
            endpoint: self.endpoint.clone(),
            session_id: self.session_id,
            device_index: self.device_index,
            responder: self.pending.responder(),
        })
    }

    /// Install a connection opened by [`wasm_connect`] and publish its handshake results.
    #[cfg(target_family = "wasm")]
    pub(crate) fn wasm_install(&mut self, connected: WasmConnected) {
        if self.writer.is_some() {
            // A concurrent connect already installed a session; drop this one rather than
            // leaking two writers for the same device.
            return;
        }
        let _ = self.settings.set(connected.settings);
        let _ = self.device_count.set(connected.device_count);
        self.writer = Some(connected.writer);
    }

    /// Hand whatever's currently buffered to the writer task as one batch (the writer
    /// serializes it off the runner thread). No-op when the buffer is empty; otherwise opens
    /// the connection first if it isn't already up.
    pub fn flush(&mut self) {
        if self.batch.is_empty() {
            return;
        }
        self.ensure_connected();
        log::trace!("Flush session: {}", self.session_id);
        let batch = self.batch.take();
        let writer = self
            .writer
            .as_ref()
            .expect("writer is set by ensure_connected");
        writer.send(&self.runtime, batch);
    }
}

impl Drop for RemoteService {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;

        // If we never connected, there's no server-side session to close and no writer to
        // drain — whatever was buffered never had a connection to go out on, so just drop it.
        if self.writer.is_none() {
            return;
        }

        // Best-effort teardown: append Close to whatever's still buffered and let the
        // writer drain + flush it before we join the task, so the runtime isn't torn down
        // mid-send. Serialization happens in the writer task now, so Drop can't panic on it.
        self.batch.push(RemoteMessage::Close(self.session_id));
        let batch = self.batch.take();
        let writer = self
            .writer
            .as_mut()
            .expect("writer present (checked above)");
        writer.shutdown(&self.runtime, Some(batch));
    }
}
