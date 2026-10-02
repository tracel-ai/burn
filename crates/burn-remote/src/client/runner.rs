use super::{ConnectError, RemoteChannel, RemoteClient, service};
use crate::shared::{LocalTransferId, TaskResponseContent, TensorRemote, TransferCapability};
use crate::{PeerAddr, PeerId};
use burn_backend::{
    DeviceId, DeviceOps, ExecutionError, ProfileDuration, ProfileOptions, ProfileToken,
    RouterDeviceType, StreamId, TensorData,
};
use burn_ir::TensorIr;
use burn_router::{MultiBackendBridge, RouterClient, RouterTensor, get_client};
use burn_std::DeviceSettings;
use burn_std::{backtrace::BackTrace, future::DynFut};
use std::sync::Mutex;

use service::RemoteEndpoint;

// It is very important to block on any request made via the service, since ordering is
// crucial when registering operations or creating tensors. The `DeviceHandle` queue
// preserves submission order, so `submit` is sufficient for cheap fire-and-forget ops; we
// only `submit_blocking` for paths that need to read the service's response.
impl RouterClient for RemoteClient {
    type Device = RemoteDevice;

    fn register_op(&self, op: burn_ir::OperationIr) {
        let stream_id = StreamId::current();
        // Device ids in the op's payload are *client* remote device ids; rewrite them to
        // server-local device indices the server can resolve to its own backend devices. Applies
        // to every op — only ops that actually carry device ids are rewritten.
        let op = self.resolve_devices(op);
        self.handle.submit(move |s| s.register_op(stream_id, op));
    }

    fn read_tensor_async(
        &self,
        tensor: burn_ir::TensorIr,
    ) -> DynFut<Result<TensorData, ExecutionError>> {
        // Issue the request synchronously so ordering is preserved relative to subsequent
        // submissions; the returned future just awaits the server's response.
        let stream_id = StreamId::current();
        let rx = self
            .handle
            .submit_blocking(move |s| s.read_tensor(stream_id, tensor))
            .expect("Service call failed");

        Box::pin(async move {
            match rx.await {
                Ok(TaskResponseContent::ReadTensor(res)) => res,
                Ok(_) => panic!("Invalid response type for ReadTensor"),
                Err(e) => Err(ExecutionError::Generic {
                    reason: format!("Failed to read tensor: {e:?}"),
                    backtrace: BackTrace::capture(),
                }),
            }
        })
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
        self.handle
            .submit_blocking(|s| s.sync(stream_id))
            .expect("Service call failed")
    }

    fn profile_start(&self) -> Result<Option<ProfileToken>, ExecutionError> {
        let stream_id = StreamId::current();
        self.handle
            .submit_blocking(move |s| s.profile_start(stream_id))
            .expect("Service call failed")
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
        Ok(self
            .handle
            .submit_blocking(move |s| s.profile_end(token, options))
            .expect("Service call failed"))
    }

    /// Told to the server rather than closed: the close is a blocking round
    /// trip whose measurement nobody is left to read, and an open window
    /// costs the server's backend something until it hears.
    fn profile_abandon(&self, token: ProfileToken) {
        self.handle.submit(move |s| s.profile_abandon(token));
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
            .expect("Service call failed")
    }

    fn flush(&self) {
        self.handle.submit_blocking(|s| s.flush()).unwrap();
    }

    fn register_and_execute_graph(
        &self,
        graph_id: burn_ir::GraphId,
        relative_graph: Vec<burn_ir::OperationIr>,
        bindings: burn_ir::GraphBindings,
    ) {
        let stream_id = StreamId::current();
        let relative_graph = relative_graph
            .into_iter()
            .map(|op| self.resolve_devices(op))
            .collect();
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
    /// Rewrite the device ids carried by an op so the server can resolve them.
    ///
    /// This runs for every op, but only ops that carry device ids (currently the collective ops)
    /// are affected. On the client, the participating devices are identified by their *remote*
    /// device ids, whose `index_id` is this process's registry index for `address` + device index.
    /// The server cannot resolve that index, so we translate each id to the device's position on
    /// the server (in `index_id`, with `type_id` 0), which the server resolves to the backend id
    /// of the device it hosts there.
    ///
    /// Only same-server collectives are supported for now: every participating device must live
    /// on the same address as the tensor's device. A cross-server group panics with a clear
    /// message rather than silently reducing the wrong devices.
    fn resolve_devices(&self, mut op: burn_ir::OperationIr) -> burn_ir::OperationIr {
        use burn_ir::{DistributedOperationIr, OperationIr};

        if let OperationIr::Distributed(DistributedOperationIr::AllReduce(desc)) = &mut op {
            let local_peer = self.device.peer_id();
            for id in desc.device_ids.iter_mut() {
                let registered = service::registered_device(id.index_id as u32).expect(
                    "an all_reduce device must be a registered remote device on this process",
                );
                assert_eq!(
                    registered.endpoint.peer_id(),
                    local_peer,
                    "cross-peer all_reduce is not supported yet: the tensor is on `{local_peer}` \
                     but the collective includes a device on `{}`",
                    registered.endpoint.peer_id(),
                );
                id.type_id = 0;
                id.index_id = registered.device_index as u16;
            }
            log::trace!("All-reduce on {:?} ({local_peer}): {desc:?}", self.device);
        }

        op
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
    /// already. A device whose session ended is not reused: a new id replaces it, with no session.
    pub(crate) fn register(endpoint: RemoteEndpoint, device_index: usize) -> Self {
        let device_index = device_index as u32;
        let id = service::register_endpoint(endpoint.clone(), device_index);
        Self {
            endpoint,
            device_index,
            id,
        }
    }

    /// [`register`](Self::register), then open its session, or confirm the one already open.
    /// A confirmation that finds the server gone leaves the device's session ended.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn open(
        endpoint: RemoteEndpoint,
        device_index: usize,
    ) -> Result<Self, ConnectError> {
        let device = Self::register(endpoint, device_index);
        get_client::<RemoteChannel>(&device).connect()?;
        Ok(device)
    }

    /// The browser's [`open`](Self::open).
    #[cfg(target_family = "wasm")]
    pub(crate) async fn open_async(
        endpoint: RemoteEndpoint,
        device_index: usize,
    ) -> Result<Self, ConnectError> {
        let device = Self::register(endpoint, device_index);
        get_client::<RemoteChannel>(&device).connect_async().await?;
        Ok(device)
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

    /// Whether this device's session has ended, as when its server restarted. Its tensors are
    /// gone with it; a new connect gives a new device.
    pub fn session_ended(&self) -> bool {
        service::session_ended(self.id)
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

static TRANSFER_COUNTER: Mutex<Option<LocalTransferId>> = Mutex::new(None);

/// Allocate the next process-unique [`LocalTransferId`] for a same-peer transfer.
///
/// The id keys the server's transfer rendezvous (`local_comm` / `external_comm`), so two
/// transfers that are ever in flight at the same time MUST get distinct ids; otherwise a `take`
/// can pick up the wrong (or an overwritten) exposed primitive and its peer hangs forever. The
/// counter is incremented **in place** in the static; `LocalTransferId` is `Copy`, so the
/// earlier `transfer_counter.unwrap()` copied the value out and incremented a throwaway local,
/// leaving every transfer after the first sharing id 1 (harmless sequentially, a deadlock under
/// concurrency).
fn get_next_transfer_id() -> LocalTransferId {
    let mut transfer_counter = TRANSFER_COUNTER.lock().unwrap();
    match transfer_counter.as_mut() {
        Some(id) => {
            id.next();
            *id
        }
        None => {
            let id = LocalTransferId::from(0);
            *transfer_counter = Some(id);
            id
        }
    }
}

impl RemoteTensorHandle {
    /// Move the tensor to `target_device`, picking the cheapest path.
    ///
    /// When the source and target live on the **same** server (same address, different device
    /// index), the data never leaves the process — see [`change_backend_local`]. Otherwise we
    /// fall back to the cross-server path that streams the data server-to-server without the
    /// client ever seeing it.
    pub(crate) fn change_backend(self, target_device: &RemoteDevice) -> Self {
        // Generations of one device share a peer, so a move between a dead session and a live
        // one would take the same-server path and wait forever on the side that is gone.
        for (side, device) in [("from", &self.client.device), ("to", target_device)] {
            assert!(
                !device.session_ended(),
                "Cannot move a tensor {side} a remote device whose session has ended; its \
                 tensors are gone with it. Connect again with `Device::remote_options`."
            );
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

    /// Same-host transfer: hand the device-resident primitive from the source session to the
    /// target session on the same server, which moves it with the inner backend's `to_device`.
    /// No host round-trip.
    fn change_backend_local(mut self, target_device: &RemoteDevice) -> Self {
        // The stream of the calling user thread: the source reads the tensor back on the
        // stream that produced it, and the target registers the result on the stream that
        // will consume it — same contract as the regular op/tensor registration paths.
        let stream_id = StreamId::current();
        let transfer_id = get_next_transfer_id();
        let tensor = self.tensor.clone();
        self.client.handle.submit(move |s| {
            s.expose_tensor_local(stream_id, tensor, transfer_id);
        });
        // Force the expose (and the ops producing the tensor) onto the wire now — same reason
        // as the cross-server path below.
        self.client.handle.flush_queue();

        let target_client = get_client::<RemoteChannel>(target_device);
        let new_id = service::new_tensor_id();
        target_client.handle.submit(move |s| {
            s.register_tensor_local(stream_id, transfer_id, new_id);
        });
        target_client.handle.flush_queue();

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
        self.client.handle.submit(move |s| {
            s.expose_tensor_remote(stream_id, tensor, 1, capability, target);
        });
        // `submit` only enqueues the closure on the device-runner queue; the runner
        // wouldn't drain it until 32 ops accumulated. `flush_queue` forces the runner
        // thread to run the closure now, and `expose_tensor_remote` itself flushes the
        // service batch onto the wire — so the source server receives the expose before
        // the target server starts trying to download.
        self.client.handle.flush_queue();

        let target_client = get_client::<RemoteChannel>(target_device);

        let peer = self.client.device.peer_addr();
        let new_id = service::new_tensor_id();
        target_client.handle.submit(move |s| {
            s.register_tensor_remote(stream_id, TensorRemote { capability, peer }, new_id);
        });
        // Same as the source side: drain the closure queue so it runs now and
        // `register_tensor_remote` flushes the registration onto the target's wire.
        target_client.handle.flush_queue();

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
