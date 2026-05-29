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
    str::FromStr,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::sync::{Mutex as AsyncMutex, oneshot};

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

/// All the state owned by the device-runner thread for a single remote device.
///
/// `RemoteService` lives behind a [`DeviceHandle`](burn_backend::DeviceHandle); every call
/// from the `RunnerClient` shim hops onto the runner thread via `submit` /
/// `submit_blocking`, so the service has exclusive access to the connection and the
/// callback map without any locking on its own state.
///
/// All tokio work — connecting, sending bytes, awaiting responses, spawning the
/// response-demux task — happens inside the runtime owned by this struct. The caller never
/// sees a runtime handle.
pub struct RemoteService<C: ProtocolClient> {
    runtime: tokio::runtime::Runtime,
    stream_request: C::Channel,
    pending: PendingMap,
    settings: Arc<OnceLock<DeviceSettings>>,
    request_counter: RequestId,
    session_id: SessionId,
    closed: bool,
}

impl<C: ProtocolClient> DeviceService for RemoteService<C> {
    fn init(device_id: DeviceId) -> Self {
        let id = device_id.index_id as u32;
        let address_str = id_to_address(id)
            .unwrap_or_else(|| panic!("No address registered for device id {device_id}"));
        let address = Address::from_str(&address_str)
            .unwrap_or_else(|_| panic!("Could not parse registered address `{address_str}`"));

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .expect("Can build tokio runtime for remote service");

        let session_id = SessionId::new();

        // Connect synchronously so failures surface here rather than on the first op,
        // and the response-demux task can be spawned with an already-open stream.
        log::info!("Connecting to {address} ...");
        let (mut stream_request, mut stream_response) = runtime.block_on(async {
            let request = C::connect(address.clone(), "request")
                .await
                .unwrap_or_else(|err| {
                    panic!(
                        "Failed to open remote 'request' channel to {address}: {err:?}. \
                     Is a `burn-remote` server running at that address?"
                    )
                });
            let response = C::connect(address.clone(), "response")
                .await
                .unwrap_or_else(|err| {
                    panic!(
                        "Failed to open remote 'response' channel to {address}: {err:?}. \
                         Is a `burn-remote` server running at that address?"
                    )
                });
            (request, response)
        });

        // Send the session-init handshake on both streams and wait for the device settings.
        let init_bytes: bytes::Bytes = rmp_serde::to_vec(&Task::Init(session_id))
            .expect("Can serialize Task::Init")
            .into();

        let device_settings: DeviceSettings = runtime
            .block_on(async {
                stream_request
                    .send(Message::new(init_bytes.clone()))
                    .await?;
                stream_response.send(Message::new(init_bytes)).await?;

                let msg = stream_response
                    .recv()
                    .await?
                    .expect("Server disconnected during initialization");

                let response: TaskResponse = rmp_serde::from_slice(&msg.data)
                    .expect("Can deserialize init handshake payload");

                match response.content {
                    TaskResponseContent::Init(settings) => Ok::<_, C::Error>(settings),
                    other => panic!("Expected Init response, got {other:?}"),
                }
            })
            .unwrap_or_else(|err| {
                panic!("Failed to initialize remote session at {address}: {err:?}")
            });

        // Publish settings to the shared cell so `RemoteDevice::defaults` can see them.
        let cell = settings_cell(id);
        let _ = cell.set(device_settings);

        let pending: PendingMap = Arc::new(AsyncMutex::new(HashMap::new()));

        // Response-demux task: each TaskResponse is routed to its pending callback by
        // RequestId. Lives on the service runtime; dies when the runtime drops.
        let pending_clone = pending.clone();
        runtime.spawn(async move {
            loop {
                match stream_response.recv().await {
                    Ok(Some(msg)) => {
                        let response: TaskResponse = match rmp_serde::from_slice(&msg.data) {
                            Ok(r) => r,
                            Err(err) => {
                                log::error!("Failed to deserialize remote response: {err:?}");
                                continue;
                            }
                        };
                        let tx = pending_clone.lock().await.remove(&response.id);
                        match tx {
                            Some(tx) => {
                                // Receiver dropped is fine (caller no longer cares).
                                let _ = tx.send(response.content);
                            }
                            None => {
                                log::warn!("No pending callback for response id {:?}", response.id)
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

        Self {
            runtime,
            stream_request,
            pending,
            settings: cell,
            request_counter: 0,
            session_id,
            closed: false,
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

impl<C: ProtocolClient> RemoteService<C> {
    /// Send a single op to the server, tagged with the stream it was issued on.
    ///
    /// Each op crosses the wire on its own — batching across streams would lose stream
    /// identity, and any same-stream batching is the server backend's responsibility.
    pub fn register_op(&mut self, stream_id: StreamId, op: OperationIr) {
        self.send_compute(ComputeTask::RegisterOperation(stream_id, op));
    }

    pub fn register_tensor(&mut self, stream_id: StreamId, id: TensorId, data: TensorData) {
        self.send_compute(ComputeTask::RegisterTensor(stream_id, id, data));
    }

    pub fn register_tensor_remote(&mut self, tensor: TensorRemote, new_id: TensorId) {
        self.send_compute(ComputeTask::RegisterTensorRemote(tensor, new_id));
    }

    pub fn expose_tensor_remote(
        &mut self,
        tensor: TensorIr,
        count: u32,
        transfer_id: TensorTransferId,
    ) {
        self.send_compute(ComputeTask::ExposeTensorRemote {
            tensor,
            count,
            transfer_id,
        });
    }

    pub fn seed(&mut self, seed: u64) {
        self.send_compute(ComputeTask::Seed(seed));
    }

    /// Initiate a tensor read. The future resolves when the server response arrives.
    ///
    /// The request id rides on the task itself; the server echoes it back so the
    /// response-demux task can hand the response to the right pending callback.
    pub fn read_tensor(
        &mut self,
        stream_id: StreamId,
        tensor: TensorIr,
    ) -> oneshot::Receiver<TaskResponseContent> {
        let request_id = self.next_request_id();
        let rx = self.register_callback(request_id);
        self.send_compute(ComputeTask::ReadTensor(request_id, stream_id, tensor));
        rx
    }

    pub fn sync(&mut self, stream_id: StreamId) -> Result<(), ExecutionError> {
        let request_id = self.next_request_id();
        let rx = self.register_callback(request_id);
        self.send_compute(ComputeTask::SyncBackend(request_id, stream_id));
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
        let request_id = self.next_request_id();
        let rx = self.register_callback(request_id);
        self.send_compute(ComputeTask::DTypeUsage(request_id, dtype));
        match self.runtime.block_on(rx) {
            Ok(TaskResponseContent::DTypeUsage(set)) => set,
            Ok(other) => panic!("Invalid response for DTypeUsage: {other:?}"),
            Err(_) => panic!("Remote response channel closed before dtype_usage completed"),
        }
    }

    fn send_compute(&mut self, task: ComputeTask) {
        self.send_task(Task::Compute(task));
    }

    fn send_task(&mut self, task: Task) {
        let bytes: bytes::Bytes = rmp_serde::to_vec(&task).expect("Can serialize task").into();
        let runtime = &self.runtime;
        let stream = &mut self.stream_request;
        runtime
            .block_on(stream.send(Message::new(bytes)))
            .expect("Send to remote request stream failed");
    }

    fn register_callback(
        &self,
        request_id: RequestId,
    ) -> oneshot::Receiver<TaskResponseContent> {
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

        // Best-effort close on teardown. Don't panic — we may already be in the middle
        // of a graceful shutdown where the stream is gone.
        let runtime = &self.runtime;
        let stream = &mut self.stream_request;

        let close = Task::Close(self.session_id);
        if let Ok(bytes) = rmp_serde::to_vec(&close) {
            let _ = runtime.block_on(stream.send(Message::new(bytes.into())));
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
