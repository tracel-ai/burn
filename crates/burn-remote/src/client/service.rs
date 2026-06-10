use crate::shared::{
    RemoteMessage, RequestId, SessionId, Task, TaskResponse, TaskResponseContent, TensorRemote,
};
use burn_backend::{
    DTypeUsageSet, ExecutionError, TensorData,
    backend::{DeviceId, DeviceService, ServerUtilitiesHandle},
};
use burn_communication::{
    Address, CommunicationChannel, Message, ProtocolClient, external_comm::TensorTransferId,
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
use writer::SubmitWriter;

use registry::{device_count_cell, settings_cell};
pub(crate) use registry::{device_count_for, has_settings, new_tensor_id, settings_for};
pub use registry::{endpoint_to_id, id_to_endpoint};

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
/// from the `RouterClient` shim hops onto the runner thread via the device handle's
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
/// decides when to flush, [`SubmitWriter`] owns the socket and serializes frames off the
/// runner thread, and [`PendingResponses`] correlates response-producing requests with the
/// caller awaiting each reply.
///
/// All tokio work — connecting, the writer task, awaiting responses, the response-demux
/// task — happens inside the runtime owned by this struct. The caller never sees a runtime
/// handle.
pub struct RemoteService<C: ProtocolClient> {
    runtime: tokio::runtime::Runtime,
    /// Where to connect on first use. The connection is established lazily (see
    /// [`ensure_connected`](Self::ensure_connected)) rather than in [`init`](Self::init),
    /// because cubecl holds a process-global device-registry lock across `init` — opening the
    /// sockets there would serialize every remote device's setup behind that lock.
    address: Address,
    device_index: u32,
    /// Owns the request socket and serializes outgoing frames off the runner thread.
    /// `None` until the first task (or a settings read) triggers the lazy connect.
    writer: Option<SubmitWriter>,
    /// Buffers outgoing tasks and signals when a batch is ready for the wire.
    batch: OutgoingBatch,
    /// Request-id allocation + the callbacks awaiting response-producing tasks.
    pending: PendingResponses,
    /// Shared cell populated from the init handshake (read by `RemoteDevice::defaults`).
    settings: Arc<OnceLock<DeviceSettings>>,
    /// Shared cell populated from the init handshake (read by `RemoteDevice::enumerate`).
    device_count: Arc<OnceLock<u32>>,
    session_id: SessionId,
    closed: bool,
    /// The request channel (`C::Channel`) lives in the writer task, not in this struct, so
    /// nothing else carries `C`. Keep the parameter pinned to the service.
    _p: PhantomData<C>,
}

impl<C: ProtocolClient> DeviceService for RemoteService<C> {
    fn init(device_id: DeviceId) -> Self {
        let (id, address, device_index) = Self::resolve_endpoint(device_id);
        let runtime = build_runtime();
        let session_id = SessionId::new();

        // Lazy connect: `init` must return promptly. cubecl holds a process-global
        // device-registry lock across this call (to make device-handle creation atomic), so
        // doing the blocking network connect + handshake here would serialize every remote
        // device's setup behind that lock — N devices would connect strictly one at a time.
        // Instead we record the endpoint and open the sockets on the first real use, off the
        // lock and on the device-runner thread (see `ensure_connected`).
        Self {
            runtime,
            address,
            device_index,
            writer: None,
            batch: OutgoingBatch::new(FLUSH_THRESHOLD),
            pending: PendingResponses::new(),
            settings: settings_cell(id),
            device_count: device_count_cell(id),
            session_id,
            closed: false,
            _p: PhantomData,
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
fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .expect("Can build tokio runtime for remote service")
}

/// Construction helpers for [`RemoteService::init`], one per step of bringing a connection
/// up. Kept separate from the public submit-style API below.
impl<C: ProtocolClient> RemoteService<C> {
    /// Resolve a device id to its registry index, parsed network [`Address`], and the device
    /// index to select on the server.
    fn resolve_endpoint(device_id: DeviceId) -> (u32, Address, u32) {
        let id = device_id.index_id as u32;
        let (address_str, device_index) = id_to_endpoint(id)
            .unwrap_or_else(|| panic!("No endpoint registered for device id {device_id}"));
        let address = Address::from_str(&address_str)
            .unwrap_or_else(|_| panic!("Could not parse registered address `{address_str}`"));
        (id, address, device_index)
    }

    /// Open the submit and fetch channels. Done synchronously so a missing server surfaces here
    /// rather than on the first op, and the demux/writer tasks can be spawned on already-open
    /// streams.
    fn connect_streams(
        runtime: &tokio::runtime::Runtime,
        address: &Address,
    ) -> (C::Channel, C::Channel) {
        runtime.block_on(async {
            let submit = C::connect(address.clone(), "submit")
                .await
                .unwrap_or_else(|err| panic!("{}", connect_error("submit", address, &err)));
            let fetch = C::connect(address.clone(), "fetch")
                .await
                .unwrap_or_else(|err| panic!("{}", connect_error("fetch", address, &err)));
            (submit, fetch)
        })
    }

    /// Send the session-init handshake on both streams and wait for the device settings the
    /// server replies with on the response stream. Both streams carry the same `Vec<RemoteMessage>`
    /// wire format; the handshake is just a single-element batch.
    fn handshake(
        runtime: &tokio::runtime::Runtime,
        request: &mut C::Channel,
        response: &mut C::Channel,
        address: &Address,
        session_id: SessionId,
        device_index: u32,
    ) -> (DeviceSettings, u32) {
        let init_bytes: bytes::Bytes =
            rmp_serde::to_vec(&vec![RemoteMessage::Init(session_id, device_index)])
                .expect("Can serialize RemoteMessage::Init")
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
                    TaskResponseContent::Init(settings, device_count) => {
                        Ok::<_, C::Error>((settings, device_count))
                    }
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
        self.submit_task(Task::RegisterOperation(stream_id, op));
    }

    pub fn register_tensor(&mut self, stream_id: StreamId, id: TensorId, data: TensorData) {
        self.submit_task(Task::RegisterTensor(stream_id, id, data));
    }

    /// Register a tensor produced by a cross-server transfer, flushing immediately.
    ///
    /// This is a coordination point with no following barrier on this client — the caller
    /// (`change_backend`) switches to the target client right after submitting it — so we
    /// flush right away instead of relying on a later op to piggy-back the flush. Otherwise
    /// the task would sit buffered below [`FLUSH_THRESHOLD`] and the target server would
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
        transfer_id: TensorTransferId,
    ) {
        self.submit_task(Task::ExposeTensorRemote {
            stream_id,
            tensor,
            count,
            transfer_id,
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
        transfer_id: TensorTransferId,
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
        transfer_id: TensorTransferId,
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

        log::debug!(
            "Connecting to {} (device {}) ...",
            self.address,
            self.device_index
        );
        let (mut request, mut response) = Self::connect_streams(&self.runtime, &self.address);
        let (settings, device_count) = Self::handshake(
            &self.runtime,
            &mut request,
            &mut response,
            &self.address,
            self.session_id,
            self.device_index,
        );

        // Publish to the shared cells so `RemoteDevice::defaults`/`enumerate` can read them.
        let _ = self.settings.set(settings);
        let _ = self.device_count.set(device_count);

        Self::spawn_response_demux(&self.runtime, response, self.pending.responder());
        self.writer = Some(SubmitWriter::spawn::<C>(&self.runtime, request));
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

impl<C: ProtocolClient> Drop for RemoteService<C> {
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
