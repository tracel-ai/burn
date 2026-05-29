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
    collections::HashMap,
    marker::PhantomData,
    str::FromStr,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::sync::{Mutex as AsyncMutex, mpsc, oneshot};

/// Global monotonic tensor-id counter, shared across all remote clients.
///
/// Mirrors the pattern used by [`burn_fusion`]: ids are allocated cheaply on the calling
/// thread without ever needing to round-trip to the device-runner thread.
static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Allocate a fresh, process-globally unique [`TensorId`].
pub(crate) fn new_tensor_id() -> TensorId {
    TensorId::new(TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Global registry mapping a network address to a stable `u32` (the `index_id` carried by
/// `RemoteDevice` → `DeviceId`) and to the cell holding the device's settings.
///
/// The same address always returns the same id; the `OnceLock<DeviceSettings>` is shared
/// between `RemoteDevice::defaults` and `RemoteService::init` so the device can surface the
/// settings without holding a handle to the service.
struct AddressRegistry {
    next_index: u32,
    by_address: HashMap<String, u32>,
    by_index: HashMap<u32, AddressEntry>,
}

#[derive(Clone)]
struct AddressEntry {
    address: String,
    settings: Arc<OnceLock<DeviceSettings>>,
}

static REGISTRY: OnceLock<Mutex<AddressRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<AddressRegistry> {
    REGISTRY.get_or_init(|| {
        Mutex::new(AddressRegistry {
            next_index: 0,
            by_address: HashMap::new(),
            by_index: HashMap::new(),
        })
    })
}

/// Map a network address to a stable `u32` id (creating one if it's the first time we see it).
///
/// Globally stable over the lifetime of the process; calling with the same address always
/// returns the same id.
pub fn address_to_id<S: AsRef<str>>(address: S) -> u32 {
    let address = address.as_ref();
    let mut reg = registry().lock().unwrap();
    if let Some(&id) = reg.by_address.get(address) {
        return id;
    }
    let id = reg.next_index;
    reg.next_index += 1;
    reg.by_address.insert(address.to_string(), id);
    reg.by_index.insert(
        id,
        AddressEntry {
            address: address.to_string(),
            settings: Arc::new(OnceLock::new()),
        },
    );
    id
}

/// Look up the address bound to `id` by [`address_to_id`].
pub fn id_to_address(id: u32) -> Option<String> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .map(|e| e.address.clone())
}

/// Returns the device settings registered for `id`.
///
/// Panics if no [`RemoteService`] has populated them yet (i.e., the client has not been
/// initialized for this device).
pub(crate) fn settings_for(id: u32) -> DeviceSettings {
    let cell = settings_cell(id);
    *cell
        .get()
        .expect("Remote service has not been initialized for this device yet")
}

/// Returns whether the device settings cell for `id` has been populated.
///
/// Used by `RemoteDevice::defaults` to decide whether a lazy-connect is needed before
/// reading.
pub(crate) fn has_settings(id: u32) -> bool {
    let reg = registry().lock().unwrap();
    reg.by_index
        .get(&id)
        .map(|e| e.settings.get().is_some())
        .unwrap_or(false)
}

fn settings_cell(id: u32) -> Arc<OnceLock<DeviceSettings>> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .expect("Device id not registered")
        .settings
        .clone()
}

type PendingMap = Arc<AsyncMutex<HashMap<RequestId, oneshot::Sender<TaskResponseContent>>>>;

/// Flush the outgoing task buffer when this many tasks have accumulated.
///
/// This is wire-level batching only — every task in the batch still carries its own
/// [`StreamId`] and any [`RequestId`] it needs, so the server sees the exact same
/// per-task semantics it would if each task arrived in its own frame. The threshold
/// caps memory and latency for chains of fire-and-forget submits that never hit a
/// [`submit_blocking`](RemoteService::submit_blocking) barrier.
const FLUSH_THRESHOLD: usize = 32;

/// Bound on serialized request frames queued for the writer task.
///
/// `flush` enqueues here instead of sending on the socket directly. Bounded so a stalled
/// socket surfaces as backpressure on the runner thread (a `flush` blocks only when the
/// writer has fallen this many batches behind) rather than as unbounded memory growth.
const WRITE_QUEUE_CAP: usize = 16;

/// All the state owned by the device-runner thread for a single remote device.
///
/// `RemoteService` lives behind a [`DeviceHandle`](burn_backend::DeviceHandle); every call
/// from the `RunnerClient` shim hops onto the runner thread via the device handle's
/// `submit` / `submit_blocking`, so the service has exclusive access to the connection,
/// the callback map, and the outgoing task buffer without any locking on its own state.
///
/// The service mirrors that submit-style API internally: fire-and-forget calls go
/// through [`submit`](Self::submit) (push into [`task_buffer`](Self::task_buffer), flush
/// once it reaches [`FLUSH_THRESHOLD`]), and response-producing calls go through
/// [`submit_blocking`](Self::submit_blocking) (push, then flush right away so the request
/// is enqueued before we await the oneshot).
///
/// Flushing does not touch the socket directly: the batch is serialized and handed to a
/// dedicated writer task over a bounded channel ([`tx`](Self::tx)). The writer owns the
/// request channel and `await`s each send fully before pulling the next, so frames hit the
/// wire in enqueue order while the runner thread stays free to keep buffering ops instead
/// of parking on the network. The bounded channel applies backpressure: a `flush` blocks
/// the runner thread only when the writer has fallen [`WRITE_QUEUE_CAP`] batches behind.
///
/// All tokio work — connecting, the writer task, awaiting responses, the response-demux
/// task — happens inside the runtime owned by this struct. The caller never sees a runtime
/// handle.
pub struct RemoteService<C: ProtocolClient> {
    runtime: tokio::runtime::Runtime,
    /// Outgoing request frames. `flush` serializes a batch and enqueues it here; the
    /// writer task drains it in FIFO order. `Option` so `Drop` can drop the sender to
    /// signal the writer to finish. Bounded — see [`WRITE_QUEUE_CAP`].
    tx: Option<mpsc::Sender<bytes::Bytes>>,
    /// The writer task handle, joined on `Drop` so the runtime isn't torn down mid-send.
    writer: Option<tokio::task::JoinHandle<()>>,
    pending: PendingMap,
    settings: Arc<OnceLock<DeviceSettings>>,
    task_buffer: Vec<Task>,
    request_counter: RequestId,
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

        let pending: PendingMap = Arc::new(AsyncMutex::new(HashMap::new()));
        Self::spawn_response_demux(&runtime, response, pending.clone());
        let (tx, writer) = Self::spawn_writer(&runtime, request);

        Self {
            runtime,
            tx: Some(tx),
            writer: Some(writer),
            pending,
            settings: cell,
            task_buffer: Vec::with_capacity(FLUSH_THRESHOLD),
            request_counter: 0,
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
    /// [`RequestId`]. Lives on the service runtime; exits when the response stream closes.
    fn spawn_response_demux(
        runtime: &tokio::runtime::Runtime,
        mut response: C::Channel,
        pending: PendingMap,
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
                        match pending.lock().await.remove(&reply.id) {
                            // Receiver dropped is fine (caller no longer cares).
                            Some(tx) => {
                                let _ = tx.send(reply.content);
                            }
                            None => {
                                log::warn!("No pending callback for response id {:?}", reply.id)
                            }
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

    /// Spawn the writer task that owns the request channel and serializes outgoing frames.
    /// Returns the bounded sender `flush` enqueues into and the task handle joined on `Drop`.
    /// The writer awaits each send fully before pulling the next, so frames reach the wire
    /// in FIFO order without ever parking the runner thread on the network.
    fn spawn_writer(
        runtime: &tokio::runtime::Runtime,
        mut request: C::Channel,
    ) -> (mpsc::Sender<bytes::Bytes>, tokio::task::JoinHandle<()>) {
        let (tx, mut rx) = mpsc::channel::<bytes::Bytes>(WRITE_QUEUE_CAP);
        let writer = runtime.spawn(async move {
            while let Some(bytes) = rx.recv().await {
                if let Err(err) = request.send(Message::new(bytes)).await {
                    log::warn!("Remote request writer send failed: {err:?}; closing writer");
                    return;
                }
            }
        });
        (tx, writer)
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
        let request_id = self.next_request_id();
        let rx = self.register_callback(request_id);
        self.submit_blocking(Task::Compute(make_task(request_id)));
        rx
    }

    /// Append a task to the outgoing buffer; flush only if we've hit the threshold.
    ///
    /// Use this for fire-and-forget tasks. The runner thread is single-threaded, so
    /// pushing into the buffer is just a `Vec::push` — no locking, no tokio hop, no
    /// network send.
    fn submit(&mut self, task: Task) {
        self.task_buffer.push(task);
        if self.task_buffer.len() >= FLUSH_THRESHOLD {
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
        self.task_buffer.push(task);
        self.flush();
    }

    /// Serialize whatever's currently in the buffer as a single frame and hand it to the
    /// writer task. No-op when the buffer is empty.
    fn flush(&mut self) {
        if self.task_buffer.is_empty() {
            return;
        }
        let batch = std::mem::take(&mut self.task_buffer);
        let bytes: bytes::Bytes = rmp_serde::to_vec(&batch)
            .expect("Can serialize task batch")
            .into();
        // Hand the frame to the writer task. Bounded channel: this blocks the runner
        // thread only when the writer has fallen `WRITE_QUEUE_CAP` batches behind (socket
        // backpressure), never on a healthy send.
        let runtime = &self.runtime;
        let tx = self.tx.as_ref().expect("Writer channel present until drop");
        runtime
            .block_on(tx.send(bytes))
            .expect("Remote request writer task alive");
    }

    fn register_callback(&self, request_id: RequestId) -> oneshot::Receiver<TaskResponseContent> {
        let (tx, rx) = oneshot::channel();
        let pending = self.pending.clone();
        self.runtime
            .block_on(async move { pending.lock().await.insert(request_id, tx) });
        rx
    }

    fn next_request_id(&mut self) -> RequestId {
        let id = self.request_counter;
        self.request_counter += 1;
        id
    }
}

impl<C: ProtocolClient> Drop for RemoteService<C> {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;

        // Best-effort flush + close on teardown: append Close to whatever's still
        // buffered and enqueue it as one final frame. Don't panic — we may already be in
        // the middle of a graceful shutdown where the writer is gone.
        self.task_buffer.push(Task::Close(self.session_id));
        let batch = std::mem::take(&mut self.task_buffer);
        if let Ok(bytes) = rmp_serde::to_vec(&batch)
            && let Some(tx) = self.tx.as_ref()
        {
            let _ = self.runtime.block_on(tx.send(bytes.into()));
        }

        // Drop the sender so the writer's `rx.recv()` returns `None` once it has drained
        // the queue (including the Close frame above), then wait for it to finish so the
        // runtime isn't torn down mid-send.
        self.tx.take();
        if let Some(writer) = self.writer.take() {
            let _ = self.runtime.block_on(writer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_to_id() {
        let address1 = "ws://127.0.0.1:3000";
        let address2 = "ws://127.0.0.1:3001";

        let id1 = address_to_id(address1);
        let id2 = address_to_id(address2);

        assert_ne!(id1, id2);

        assert_eq!(address_to_id(address1), id1);
        assert_eq!(id_to_address(id1), Some(address1.to_string()));

        assert_eq!(address_to_id(address2), id2);
        assert_eq!(id_to_address(id2), Some(address2.to_string()));

        let unused_id = u32::MAX;
        assert_eq!(id_to_address(unused_id), None);
    }
}
