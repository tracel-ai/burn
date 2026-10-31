use super::{DeviceQueryError, RemoteChannel, RemoteClient, service};
use crate::shared::{DeviceQueryId, LocalTransferId, TensorRemote, TransferCapability};
use crate::{PeerAddr, PeerId};
use burn_backend::{
    DeviceDescription, DeviceId, DeviceOps, DeviceTelemetry, ExecutionError, ExecutionMode,
    ExecutionModeToken, ExecutionStatistics, MemoryReport, Peak, PeakError, PeakKey,
    ProfileDuration, ProfileOptions, ProfileToken, RouterDeviceType, StreamId, TensorData,
};
use burn_ir::{CapturedGraphId, TensorIr};
use burn_router::{MultiBackendBridge, RouterClient, RouterTensor, get_client};
use burn_std::{DeviceSettings, device_handle::CallError, future::DynFut};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard, PoisonError};

use service::{RemoteEndpoint, RemoteService};

// It is very important to block on any request made via the service, since ordering is
// crucial when registering operations or creating tensors. The `DeviceHandle` queue
// preserves submission order, so `submit` is sufficient for cheap fire-and-forget ops; we
// only `submit_blocking` for paths that need to read the service's response.
impl RouterClient for RemoteClient {
    type Device = RemoteDevice;

    fn register_op(&self, op: burn_ir::OperationIr) {
        let stream_id = StreamId::current();
        self.handle.submit(move |s| s.register_op(stream_id, op));
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let read = self.read_tensors(vec![tensor]);
        Box::pin(async move {
            let [data] = <[TensorData; 1]>::try_from(read.await?).unwrap_or_else(|data| {
                panic!(
                    "The server answered a read of one tensor with {} values",
                    data.len()
                )
            });
            Ok(data)
        })
    }

    fn read_tensors_async(
        &self,
        tensors: Vec<TensorIr>,
    ) -> DynFut<Result<Vec<TensorData>, ExecutionError>> {
        self.read_tensors(tensors)
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let shape = data.shape().clone();
        let dtype = data.dtype();
        let id = service::new_tensor_id();

        // Fire-and-forget: the outgoing batch flushes itself once buffered data bytes (or the task
        // count) cross their threshold — see `OutgoingBatch` — so no explicit flush is needed here.
        let stream_id = StreamId::current();
        self.handle
            .submit(move |s| s.register_tensor(stream_id, id, data));

        RouterTensor::new(id, shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        let stream_id = StreamId::current();
        self.call(|s| s.sync(stream_id))
    }

    fn profile_start(&self) -> Result<Option<ProfileToken>, ExecutionError> {
        let stream_id = StreamId::current();
        self.call(move |s| s.profile_start(stream_id))
    }

    fn profile_end(
        &self,
        token: ProfileToken,
        options: ProfileOptions,
    ) -> Result<ProfileDuration, ExecutionError> {
        // The stream is the service's to name, not this thread's: it is the
        // one the window was opened on, which the service kept.
        //
        // Blocking only on the issue, so the close keeps its place among the
        // tasks around it; the measurement is awaited through the duration.
        self.handle
            .submit_blocking(move |s| s.profile_end(token, options))
            .map_err(RemoteClient::service_stopped)
    }

    /// Told to the server rather than closed: the close is a blocking round
    /// trip whose measurement nobody is left to read, and an open window
    /// costs the server's backend something until it hears.
    fn profile_abandon(&self, token: ProfileToken) {
        self.handle.submit(move |s| s.profile_abandon(token));
    }

    fn execution_mode_open(
        &self,
        mode: ExecutionMode,
    ) -> Result<Option<ExecutionModeToken>, ExecutionError> {
        let stream_id = StreamId::current();
        self.handle
            .submit_blocking(move |s| s.execution_mode_open(stream_id, mode))
            .map_err(service_failed)?
    }

    fn execution_mode_statistics(
        &self,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        self.handle
            .submit_blocking(move |s| s.execution_mode_statistics(token))
            .map_err(service_failed)?
    }

    /// An error rather than a panic when the service is gone: a guard closes its mode from
    /// `Drop`, possibly while a panic is already unwinding.
    fn execution_mode_close(
        &self,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        self.handle
            .submit_blocking(move |s| s.execution_mode_close(token))
            .map_err(service_failed)?
    }

    fn device_description(&self) -> Option<DeviceDescription> {
        self.handle
            .submit_blocking(|s| s.device_description())
            .ok()
            .flatten()
    }

    fn measure_telemetry(&self) -> DeviceTelemetry {
        self.handle
            .submit_blocking(|s| s.measure_telemetry())
            .unwrap_or_default()
    }

    fn measure_memory(&self) -> Option<MemoryReport> {
        let stream_id = StreamId::current();
        self.handle
            .submit_blocking(move |s| s.measure_memory(stream_id))
            .ok()
            .flatten()
    }

    fn measure_peaks(&self, keys: &[PeakKey]) -> Vec<Result<Peak, PeakError>> {
        let requested = keys.to_vec();
        self.handle
            .submit_blocking(move |s| s.measure_peaks(requested))
            .unwrap_or_else(|_| vec![Err(PeakError::Unsupported); keys.len()])
    }

    fn graph_prepare(&self) -> Result<(), ExecutionError> {
        let stream_id = StreamId::current();
        self.call(move |s| s.graph_prepare(stream_id))
    }

    fn graph_start_capture(&self) -> Result<(), ExecutionError> {
        let stream_id = StreamId::current();
        self.call(move |s| s.graph_start_capture(stream_id))
    }

    fn graph_stop_capture(&self) -> Result<CapturedGraphId, ExecutionError> {
        let stream_id = StreamId::current();
        self.call(move |s| s.graph_stop_capture(stream_id))
    }

    /// `Ok` once the replay is sent: the server answers no replay, and a failed one is the error
    /// of the next read or sync on this stream instead (see
    /// [`CaptureReplay`](crate::shared::Task::CaptureReplay)).
    unsafe fn graph_replay(&self, graph: CapturedGraphId) -> Result<(), ExecutionError> {
        let stream_id = StreamId::current();
        self.handle
            .submit(move |s| s.graph_replay(stream_id, graph));
        Ok(())
    }

    fn graph_release(&self, graph: CapturedGraphId) {
        self.handle.submit(move |s| s.graph_release(graph));
    }

    fn seed(&self, seed: u64) {
        self.handle.submit(move |s| s.seed(seed));
    }

    fn create_empty_handle(&self) -> burn_ir::TensorId {
        service::new_tensor_id()
    }

    fn register_alias(&self, new_id: burn_ir::TensorId, src_id: burn_ir::TensorId) {
        let stream_id = StreamId::current();
        self.handle
            .submit(move |s| s.register_alias(stream_id, new_id, src_id));
    }

    fn dtype_usage(&self, dtype: burn_std::DType) -> burn_backend::DTypeUsageSet {
        self.handle
            .submit_blocking(move |s| s.dtype_usage(dtype))
            .unwrap_or_else(|err| panic!("{}", Self::service_stopped(err)))
    }

    fn flush(&self) -> Result<(), ExecutionError> {
        self.call(|s| s.flush())
    }

    fn register_and_execute_graph(
        &self,
        graph_id: burn_ir::GraphId,
        relative_graph: Vec<burn_ir::OperationIr>,
        bindings: burn_ir::GraphBindings,
    ) {
        let stream_id = StreamId::current();
        self.handle.submit(move |s| {
            s.register_and_execute_graph(stream_id, graph_id, relative_graph, bindings)
        });
    }

    fn execute_graph(&self, graph_id: burn_ir::GraphId, bindings: burn_ir::GraphBindings) {
        let stream_id = StreamId::current();
        self.handle
            .submit(move |s| s.execute_graph(stream_id, graph_id, bindings));
    }
}

impl RemoteClient {
    /// Read several tensors in one request, issued at the call.
    fn read_tensors(
        &self,
        tensors: Vec<TensorIr>,
    ) -> DynFut<Result<Vec<TensorData>, ExecutionError>> {
        // Issue the request synchronously so ordering is preserved relative to subsequent
        // submissions; the returned future just awaits the server's response.
        let stream_id = StreamId::current();
        self.handle
            .submit_blocking(move |s| s.read_tensors(stream_id, tensors))
            .unwrap_or_else(|err| {
                let error = Self::service_stopped(err);
                Box::pin(async move { Err(error) })
            })
    }
}

impl RemoteClient {
    /// Run `call` on the service, which reports an error of its own or the service having
    /// stopped.
    fn call<T: Send>(
        &self,
        call: impl FnOnce(&mut RemoteService) -> Result<T, ExecutionError> + Send,
    ) -> Result<T, ExecutionError> {
        self.handle
            .submit_blocking(call)
            .unwrap_or_else(|err| Err(Self::service_stopped(err)))
    }

    /// The service's thread is gone, with the session it held.
    fn service_stopped(err: CallError) -> ExecutionError {
        ExecutionError::device_poisoned(format!("The remote device's service stopped: {err:?}"))
    }
}

#[derive(Clone, Debug)]
/// A remote compute device identified by its endpoint and device index.
///
/// Two RemoteDevices with the same endpoint but different indices point at distinct devices on
/// the same peer; each gets its own registry id and service connection, and transfers between
/// them take the same-peer fast path.
pub struct RemoteDevice {
    pub(crate) endpoint: RemoteEndpoint,
    /// Device index on the remote peer.
    pub(crate) device_index: u32,
    /// Local registry id for this device.
    pub(crate) id: u32,
}

impl RemoteDevice {
    /// The device registered for `endpoint` and `device_index`, which may have its session open
    /// already, or a new one once its session ended.
    pub(crate) fn register(endpoint: RemoteEndpoint, device_index: usize) -> Self {
        let device_index = device_index as u32;
        let id = service::register_endpoint(endpoint.clone(), device_index);
        Self {
            endpoint,
            device_index,
            id,
        }
    }

    /// A WebSocket device with no session opened yet, which connects on first use.
    #[cfg(feature = "websocket")]
    pub(crate) fn websocket(address: &str, device_index: usize) -> Self {
        Self::register(
            RemoteEndpoint::WebSocket {
                address: burn_communication::Address::from(address),
                credential: crate::Credential::default(),
            },
            device_index,
        )
    }

    /// An Iroh device dialed from the application's `endpoint`, which connects on first use.
    #[cfg(all(test, feature = "iroh", not(feature = "fusion")))]
    pub(crate) fn iroh(
        endpoint: &iroh::Endpoint,
        peer: iroh::EndpointAddr,
        device_index: usize,
    ) -> Self {
        Self::register(
            RemoteEndpoint::Iroh {
                node: crate::transport::iroh::node::RemoteNode::for_endpoint(endpoint)
                    .expect("one live endpoint per id"),
                peer,
                credential: crate::Credential::default(),
                app_endpoint: Some(endpoint.id()),
            },
            device_index,
        )
    }

    /// The stable identity of the compute peer.
    pub(crate) fn peer_id(&self) -> PeerId {
        self.endpoint.peer_id()
    }

    /// The peer identity plus its current dialing hints.
    pub(crate) fn peer_addr(&self) -> PeerAddr {
        self.endpoint.peer_addr()
    }

    /// The index of this device on its server.
    pub fn device_index(&self) -> usize {
        self.device_index as usize
    }

    /// Ask the server the device query `id` about this device, opening its session first if it
    /// is not open yet; the server answers with the handler registered under `id` (see
    /// `BackendServer::with_device_query`), in bytes the two sides agree on.
    ///
    /// The server runs the handler on this device's session, after the work sent before it and
    /// ahead of the work sent after.
    #[cfg(not(target_family = "wasm"))]
    pub fn query(&self, id: impl Into<DeviceQueryId>) -> Result<Vec<u8>, DeviceQueryError> {
        let client = get_client::<RemoteChannel>(self);
        client.connect().map_err(DeviceQueryError::Connect)?;
        let id = id.into();
        // The service's thread is gone only when the session went with it.
        client
            .handle
            .submit_blocking(move |s| s.device_query(id))
            .unwrap_or(Err(DeviceQueryError::SessionEnded))
    }

    /// Whether this device's session has ended, as when its server restarted.
    pub(crate) fn session_ended(&self) -> bool {
        service::session_end(self.id).has_ended()
    }

    /// The error every use of this device reports once its session has ended.
    pub(crate) fn session_error(&self) -> Option<ExecutionError> {
        service::session_end(self.id).error()
    }
}

impl PartialEq for RemoteDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RemoteDevice {}

impl Default for RemoteDevice {
    fn default() -> Self {
        #[cfg(feature = "websocket")]
        {
            let address = match std::env::var("BURN_REMOTE_ADDRESS") {
                Ok(address) => address,
                Err(_) => String::from("ws://127.0.0.1:3000"),
            };

            Self::websocket(&address, 0)
        }
        #[cfg(not(feature = "websocket"))]
        panic!(
            "RemoteDevice::default needs the `websocket` feature; connect with \
             `Device::remote_options` instead"
        )
    }
}

impl burn_std::device::Device for RemoteDevice {
    fn from_id(device_id: DeviceId) -> Self {
        assert_eq!(
            device_id.type_id,
            u16::from(RouterDeviceType::Remote),
            "invalid remote device type"
        );
        let registered = service::registered_device(device_id.index_id as u32)
            .unwrap_or_else(|| panic!("Invalid device id: {device_id}"));
        Self {
            endpoint: registered.endpoint,
            device_index: registered.device_index,
            id: device_id.index_id as u32,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: RouterDeviceType::Remote.into(),
            index_id: self.id as u16,
        }
    }
}

impl DeviceOps for RemoteDevice {
    fn defaults(&self) -> DeviceSettings {
        // Lazy-connect on first access. Callers like `Device::configure` or
        // `Device::default()`-driven dispatch can hit `defaults` before the user has
        // triggered any op, so we need to establish the session here. `connect` is
        // idempotent — a no-op once the client has been initialized for this device.
        if !service::has_settings(self.id)
            && let Err(err) = get_client::<RemoteChannel>(self).connect()
        {
            panic!(
                "Failed to open a remote session at {}: {err}",
                self.peer_addr()
            );
        }
        service::settings_for(self.id)
    }
}

pub struct RemoteBridge;

pub struct RemoteTensorHandle {
    pub(crate) client: RemoteClient,
    pub(crate) tensor: TensorIr,
}

/// Held while one call sends tasks to the sessions of `devices`, so that every session receives
/// the tasks of concurrent calls sharing it in one order: a session waiting in a transfer or a
/// collective for another session's task would otherwise wait on one queued behind its own.
pub fn session_order<'a>(
    devices: impl IntoIterator<Item = &'a RemoteDevice>,
) -> Vec<MutexGuard<'static, ()>> {
    // One per registered device, which the device registry also keeps for the process's lifetime.
    static ORDERS: Mutex<BTreeMap<u32, &'static Mutex<()>>> = Mutex::new(BTreeMap::new());
    let ids: BTreeSet<u32> = devices.into_iter().map(|device| device.id).collect();
    let orders: Vec<&'static Mutex<()>> = {
        let mut orders = ORDERS.lock().unwrap_or_else(PoisonError::into_inner);
        ids.into_iter()
            .map(|id| {
                *orders
                    .entry(id)
                    .or_insert_with(|| Box::leak(Box::default()))
            })
            .collect()
    };
    // In ascending id order, so two calls sharing sessions never each hold one the other waits on.
    orders
        .into_iter()
        .map(|order| order.lock().unwrap_or_else(PoisonError::into_inner))
        .collect()
}

impl RemoteTensorHandle {
    /// Move the tensor to `target_device`, picking the cheapest path.
    ///
    /// When the source and target live on the **same** server (same address, different device
    /// index), the data never leaves the process — see [`change_backend_local`]. Otherwise we
    /// fall back to the cross-server path that streams the data server-to-server without the
    /// client ever seeing it.
    ///
    /// A move to or from a device whose session has ended sends no transfer: its tensors are gone
    /// with it. The tensor on the target is failed by that end instead, and a read of it reports
    /// why. Only an end this client has already seen is caught.
    pub(crate) fn change_backend(self, target_device: &RemoteDevice) -> Self {
        let ended = [&self.client.device, target_device].map(RemoteDevice::session_error);
        if let Some(error) = ended.into_iter().flatten().next() {
            return self.fail_on(target_device, error);
        }
        if self.client.device.peer_id() == target_device.peer_id() {
            self.change_backend_local(target_device)
        } else {
            assert_eq!(
                self.client.device.peer_addr().is_iroh(),
                target_device.peer_addr().is_iroh(),
                "Moving a tensor between an Iroh server and a WebSocket server is not supported"
            );
            self.change_backend_remote(target_device)
        }
    }

    /// The tensor on `target_device`, failed by `error` instead of transferred. A target whose
    /// session ended takes nothing, and every use of the tensor reports that end.
    fn fail_on(mut self, target_device: &RemoteDevice, error: ExecutionError) -> Self {
        let target_client = get_client::<RemoteChannel>(target_device);
        let stream_id = StreamId::current();
        let new_id = service::new_tensor_id();
        let reason = format!("the transfer from {:?} failed: {error}", self.client.device);
        target_client
            .handle
            .submit(move |s| s.register_failure(stream_id, new_id, reason));

        // `into_ir()` suppressed the handle's drop, expecting the transfer to consume the source.
        if matches!(self.tensor.status, burn_ir::TensorStatus::ReadWrite) {
            self.client
                .register_op(burn_ir::OperationIr::Drop(self.tensor.clone()));
        }
        self.tensor.id = new_id;
        self.client = target_client;
        self
    }

    /// Same-host transfer: hand the device-resident primitive from the source session to the
    /// target session on the same server, which moves it with the inner backend's `to_device`.
    /// No host round-trip.
    fn change_backend_local(mut self, target_device: &RemoteDevice) -> Self {
        // The stream of the calling user thread: the source reads the tensor back on the
        // stream that produced it, and the target registers the result on the stream that
        // will consume it — same contract as the regular op/tensor registration paths.
        let stream_id = StreamId::current();
        let transfer_id = LocalTransferId::random();
        let tensor = self.tensor.clone();
        let target_client = get_client::<RemoteChannel>(target_device);
        let order = session_order([&self.client.device, target_device]);
        self.client.handle.submit(move |s| {
            s.expose_tensor_local(stream_id, tensor, transfer_id);
        });
        // Force the expose (and the ops producing the tensor) onto the wire now — same reason
        // as the cross-server path below.
        self.client.handle.flush_queue();

        let new_id = service::new_tensor_id();
        target_client.handle.submit(move |s| {
            s.register_tensor_local(stream_id, transfer_id, new_id);
        });
        target_client.handle.flush_queue();
        drop(order);

        self.tensor.id = new_id;
        self.client = target_client;

        self
    }

    /// Changes the backend of the tensor via a dWebSocket.
    /// We ask the original server to expose the tensor, then ask the target server to fetch
    /// the tensor. The target server will open a new network connection to the original server
    /// to download the data.
    /// This way the client never sees the tensor's data, and we avoid a bottleneck.
    fn change_backend_remote(mut self, target_device: &RemoteDevice) -> Self {
        // See `change_backend_local`: carry the calling thread's stream so the source readback
        // and the target registration land on the client streams, not arbitrary server threads.
        let stream_id = StreamId::current();
        let capability = TransferCapability::random();
        let tensor = self.tensor.clone();
        let target = target_device.peer_id();
        let target_client = get_client::<RemoteChannel>(target_device);
        let order = session_order([&self.client.device, target_device]);
        self.client.handle.submit(move |s| {
            s.expose_tensor_remote(stream_id, tensor, 1, capability, target);
        });
        // `submit` only enqueues the closure on the device-runner queue; the runner
        // wouldn't drain it until 32 ops accumulated. `flush_queue` forces the runner
        // thread to run the closure now, and `expose_tensor_remote` itself flushes the
        // service batch onto the wire — so the source server receives the expose before
        // the target server starts trying to download.
        self.client.handle.flush_queue();

        let peer = self.client.device.peer_addr();
        let new_id = service::new_tensor_id();
        target_client.handle.submit(move |s| {
            s.register_tensor_remote(stream_id, TensorRemote { capability, peer }, new_id);
        });
        // Same as the source side: drain the closure queue so it runs now and
        // `register_tensor_remote` flushes the registration onto the target's wire.
        target_client.handle.flush_queue();
        drop(order);

        self.tensor.id = new_id;
        self.client = target_client;

        self
    }
}

impl MultiBackendBridge for RemoteBridge {
    type TensorHandle = RemoteTensorHandle;
    type Device = RemoteDevice;

    fn change_backend_float(
        tensor: Self::TensorHandle,
        _shape: burn_backend::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }

    fn change_backend_int(
        tensor: Self::TensorHandle,
        _shape: burn_backend::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }

    fn change_backend_bool(
        tensor: Self::TensorHandle,
        _shape: burn_backend::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }
}

/// The remote service could not run a call: its thread panicked, or is gone.
fn service_failed(err: burn_std::device_handle::CallError) -> ExecutionError {
    ExecutionError::with_context(format!("the remote service failed to run a call: {err:?}"))
}
