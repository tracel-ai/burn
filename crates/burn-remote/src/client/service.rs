use crate::shared::{
    ComputeTask, RequestId, SessionId, Task, TaskResponse, TaskResponseContent, TensorRemote,
};
use burn_backend::{
    DTypeUsageSet, ExecutionError, TensorData,
    backend::{DeviceId, DeviceService, ServerUtilitiesHandle},
};
use burn_communication::{
    Address, CommunicationChannel, Message, ProtocolClient, data_service::TensorTransferId,
};
use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_std::{DType, DeviceSettings, backtrace::BackTrace, id::StreamId};
use std::{
    marker::PhantomData,
    str::FromStr,
    sync::{Arc, OnceLock},
};
use tokio::sync::oneshot;

mod batch;
mod pending;
mod registry;
mod writer;

use batch::OutgoingBatch;
use pending::{PendingResponses, Responder};
use writer::RequestWriter;

pub use registry::{address_to_id, id_to_address};
pub(crate) use registry::{has_settings, new_tensor_id, settings_for};
use registry::settings_cell;

/// Flush the outgoing task buffer when this many tasks have accumulated.
///
/// This is wire-level batching only — every task in the batch still carries its own
/// [`StreamId`] and any [`RequestId`] it needs, so the server sees the exact same
/// per-task semantics it would if each task arrived in its own frame. The threshold
/// caps memory and latency for chains of fire-and-forget submits that never hit a
/// [`submit_blocking`](RemoteService::submit_blocking) barrier.
const FLUSH_THRESHOLD: usize = 32;

/// All the state owned by the device-runner thread for a single remote device.
///
/// `RemoteService` lives behind a [`DeviceHandle`](burn_backend::DeviceHandle); every call
/// from the `RunnerClient` shim hops onto the runner thread via the device handle's
/// `submit` / `submit_blocking`, so the service has exclusive access to the connection,
/// the callback map, and the outgoing task buffer without any locking on its own state.
///
/// The service mirrors that submit-style API internally: fire-and-forget calls go
/// through [`submit`](Self::submit) (push onto the [`batch`](Self::batch), flush once it
/// reaches [`FLUSH_THRESHOLD`]), and response-producing calls go through
/// [`submit_blocking`](Self::submit_blocking) (push, then flush right away so the request
/// is enqueued before we await the oneshot).
///
/// The moving parts are split into focused types: [`OutgoingBatch`] buffers tasks and
/// decides when to flush, [`RequestWriter`] owns the socket and serializes frames off the
/// runner thread, and [`PendingResponses`] correlates response-producing requests with the
/// caller awaiting each reply.
///
/// All tokio work — connecting, the writer task, awaiting responses, the response-demux
/// task — happens inside the runtime owned by this struct. The caller never sees a runtime
/// handle.
pub struct RemoteService<C: ProtocolClient> {
    runtime: tokio::runtime::Runtime,
    /// Owns the request socket and serializes outgoing frames off the runner thread.
    writer: RequestWriter,
    /// Buffers outgoing tasks and signals when a batch is ready for the wire.
    batch: OutgoingBatch,
    /// Request-id allocation + the callbacks awaiting response-producing tasks.
    pending: PendingResponses,
    settings: Arc<OnceLock<DeviceSettings>>,
    session_id: SessionId,
    closed: bool,
    /// The request channel (`C::Channel`) lives in the writer task, not in this struct, so
    /// nothing else carries `C`. Keep the parameter pinned to the service.
    _p: PhantomData<C>,
}

impl<C: ProtocolClient> DeviceService for RemoteService<C> {
    fn init(device_id: DeviceId) -> Self {
        let (id, address) = Self::resolve_address(device_id);
        let runtime = build_runtime();
        let session_id = SessionId::new();

        log::info!("Connecting to {address} ...");
        let (mut request, mut response) = Self::connect_streams(&runtime, &address);
        let settings = Self::handshake(&runtime, &mut request, &mut response, &address, session_id);

        // Publish settings to the shared cell so `RemoteDevice::defaults` can see them.
        let cell = settings_cell(id);
        let _ = cell.set(settings);

        let pending = PendingResponses::new();
        Self::spawn_response_demux(&runtime, response, pending.responder());
        let writer = RequestWriter::spawn::<C>(&runtime, request);

        Self {
            runtime,
            writer,
            batch: OutgoingBatch::new(FLUSH_THRESHOLD),
            pending,
            settings: cell,
            session_id,
            closed: false,
            _p: PhantomData,
        }
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        // DeviceSettings is `Copy`, so we publish a snapshot.
        let settings = *self
            .settings
            .get()
            .expect("DeviceSettings populated during init()");
        Arc::new(settings)
    }
}

/// Build the multi-threaded tokio runtime that hosts the connection, the writer task, and
/// the response-demux task. IO is enabled for the websocket; the runner thread enters it
/// only via `block_on`.
fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .expect("Can build tokio runtime for remote service")
}

/// Construction helpers for [`RemoteService::init`], one per step of bringing a connection
/// up. Kept separate from the public submit-style API below.
impl<C: ProtocolClient> RemoteService<C> {
    /// Resolve a device id to its registry index and parsed network [`Address`].
    fn resolve_address(device_id: DeviceId) -> (u32, Address) {
        let id = device_id.index_id as u32;
        let address_str = id_to_address(id)
            .unwrap_or_else(|| panic!("No address registered for device id {device_id}"));
        let address = Address::from_str(&address_str)
            .unwrap_or_else(|_| panic!("Could not parse registered address `{address_str}`"));
        (id, address)
    }

    /// Open the request and response channels. Done synchronously so a missing server
    /// surfaces here rather than on the first op, and the demux/writer tasks can be spawned
    /// on already-open streams.
    fn connect_streams(
        runtime: &tokio::runtime::Runtime,
        address: &Address,
    ) -> (C::Channel, C::Channel) {
        runtime.block_on(async {
            let request = C::connect(address.clone(), "request")
                .await
                .unwrap_or_else(|err| panic!("{}", connect_error("request", address, &err)));
            let response = C::connect(address.clone(), "response")
                .await
                .unwrap_or_else(|err| panic!("{}", connect_error("response", address, &err)));
            (request, response)
        })
    }

    /// Send the session-init handshake on both streams and wait for the device settings the
    /// server replies with on the response stream. Both streams carry the same `Vec<Task>`
    /// wire format; the handshake is just a single-element batch.
    fn handshake(
        runtime: &tokio::runtime::Runtime,
        request: &mut C::Channel,
        response: &mut C::Channel,
        address: &Address,
        session_id: SessionId,
    ) -> DeviceSettings {
        let init_bytes: bytes::Bytes = rmp_serde::to_vec(&vec![Task::Init(session_id)])
            .expect("Can serialize Task::Init")
            .into();

        runtime
            .block_on(async {
                request.send(Message::new(init_bytes.clone())).await?;
                response.send(Message::new(init_bytes)).await?;

                let msg = response
                    .recv()
                    .await?
                    .expect("Server disconnected during initialization");
                let reply: TaskResponse = rmp_serde::from_slice(&msg.data)
                    .expect("Can deserialize init handshake payload");

                match reply.content {
                    TaskResponseContent::Init(settings) => Ok::<_, C::Error>(settings),
                    other => panic!("Expected Init response, got {other:?}"),
                }
            })
            .unwrap_or_else(|err| {
                panic!("Failed to initialize remote session at {address}: {err:?}")
            })
    }

    /// Spawn the response-demux task: route each [`TaskResponse`] to its pending callback by
    /// [`RequestId`] via the [`Responder`]. Lives on the service runtime; exits when the
    /// response stream closes.
    fn spawn_response_demux(
        runtime: &tokio::runtime::Runtime,
        mut response: C::Channel,
        responder: Responder,
    ) {
        runtime.spawn(async move {
            loop {
                match response.recv().await {
                    Ok(Some(msg)) => {
                        let reply: TaskResponse = match rmp_serde::from_slice(&msg.data) {
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
                        return;
                    }
                    Err(err) => {
                        log::warn!("Remote response stream error: {err:?}");
                        return;
                    }
                }
            }
        });
    }
}

/// Actionable panic message for a failed channel connect.
fn connect_error<E: std::fmt::Debug>(route: &str, address: &Address, err: &E) -> String {
    format!(
        "Failed to open remote '{route}' channel to {address}: {err:?}. \
         Is a `burn-remote` server running at that address?"
    )
}

impl<C: ProtocolClient> RemoteService<C> {
    /// Buffer a fire-and-forget op. The buffer is flushed automatically once it reaches
    /// [`FLUSH_THRESHOLD`] entries.
    pub fn register_op(&mut self, stream_id: StreamId, op: OperationIr) {
        self.submit_compute(ComputeTask::RegisterOperation(stream_id, op));
    }

    pub fn register_tensor(&mut self, stream_id: StreamId, id: TensorId, data: TensorData) {
        self.submit_compute(ComputeTask::RegisterTensor(stream_id, id, data));
    }

    /// Optimistic: the cross-server transfer task is buffered like any other. The next
    /// op on this client will piggy-back the flush, and reads/syncs always flush before
    /// they wait — so as long as a barrier follows shortly, the source server has had
    /// time to expose the tensor before the target server starts downloading.
    pub fn register_tensor_remote(&mut self, tensor: TensorRemote, new_id: TensorId) {
        self.submit_compute(ComputeTask::RegisterTensorRemote(tensor, new_id));
    }

    pub fn expose_tensor_remote(
        &mut self,
        tensor: TensorIr,
        count: u32,
        transfer_id: TensorTransferId,
    ) {
        self.submit_compute(ComputeTask::ExposeTensorRemote {
            tensor,
            count,
            transfer_id,
        });
    }

    pub fn seed(&mut self, seed: u64) {
        self.submit_compute(ComputeTask::Seed(seed));
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
        self.submit_request(|id| ComputeTask::ReadTensor(id, stream_id, tensor))
    }

    pub fn sync(&mut self, stream_id: StreamId) -> Result<(), ExecutionError> {
        let rx = self.submit_request(|id| ComputeTask::SyncBackend(id, stream_id));
        match self.runtime.block_on(rx) {
            Ok(TaskResponseContent::SyncBackend(res)) => res,
            Ok(other) => panic!("Invalid response for SyncBackend: {other:?}"),
            Err(_) => Err(ExecutionError::Generic {
                reason: "Remote response channel closed before sync completed".into(),
                backtrace: BackTrace::capture(),
            }),
        }
    }

    pub fn dtype_usage(&mut self, dtype: DType) -> DTypeUsageSet {
        let rx = self.submit_request(|id| ComputeTask::DTypeUsage(id, dtype));
        match self.runtime.block_on(rx) {
            Ok(TaskResponseContent::DTypeUsage(set)) => set,
            Ok(other) => panic!("Invalid response for DTypeUsage: {other:?}"),
            Err(_) => panic!("Remote response channel closed before dtype_usage completed"),
        }
    }

    /// Buffer a fire-and-forget compute task. Thin wrapper over [`submit`](Self::submit)
    /// that wraps the task in [`Task::Compute`].
    fn submit_compute(&mut self, task: ComputeTask) {
        self.submit(Task::Compute(task));
    }

    /// Issue a response-producing compute task: allocate its [`RequestId`], register the
    /// pending callback, and flush immediately so it's enqueued before the caller awaits
    /// the returned receiver. `make_task` builds the task from the freshly allocated id.
    fn submit_request(
        &mut self,
        make_task: impl FnOnce(RequestId) -> ComputeTask,
    ) -> oneshot::Receiver<TaskResponseContent> {
        let request_id = self.pending.next_id();
        let rx = self.pending.register(request_id);
        self.submit_blocking(Task::Compute(make_task(request_id)));
        rx
    }

    /// Append a task to the outgoing buffer; flush only once it hits the threshold.
    ///
    /// Use this for fire-and-forget tasks. The runner thread is single-threaded, so
    /// pushing into the buffer is just a `Vec::push` — no locking, no tokio hop, no
    /// network send.
    fn submit(&mut self, task: Task) {
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
    fn submit_blocking(&mut self, task: Task) {
        self.batch.push(task);
        self.flush();
    }

    /// Serialize whatever's currently buffered as a single frame and hand it to the writer
    /// task. No-op when the buffer is empty.
    fn flush(&mut self) {
        if self.batch.is_empty() {
            return;
        }
        let frame = serialize_batch(self.batch.take());
        self.writer.send(&self.runtime, frame);
    }
}

/// Serialize a batch of tasks into a single wire frame.
fn serialize_batch(batch: Vec<Task>) -> bytes::Bytes {
    rmp_serde::to_vec(&batch)
        .expect("Can serialize task batch")
        .into()
}

impl<C: ProtocolClient> Drop for RemoteService<C> {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;

        // Best-effort teardown: append Close to whatever's still buffered and let the
        // writer drain + flush it before we join the task, so the runtime isn't torn down
        // mid-send. Serialization can't realistically fail, but don't panic in Drop if it
        // does — pass `None` and still join.
        self.batch.push(Task::Close(self.session_id));
        let frame = rmp_serde::to_vec(&self.batch.take()).ok().map(Into::into);
        self.writer.shutdown(&self.runtime, frame);
    }
}
