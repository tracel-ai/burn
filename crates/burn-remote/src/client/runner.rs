use super::{RemoteChannel, RemoteClient, service};
use crate::shared::{TaskResponseContent, TensorRemote};
use burn_backend::{DeviceId, DeviceOps, ExecutionError, StreamId, TensorData};
use burn_communication::{Address, ProtocolClient, external_comm::TensorTransferId};
use burn_ir::TensorIr;
use burn_router::{MultiBackendBridge, RouterClient, RouterTensor, get_client};
use burn_std::DeviceSettings;
use burn_std::{backtrace::BackTrace, future::DynFut};
use std::sync::Mutex;
use std::{marker::PhantomData, str::FromStr};

pub use service::{endpoint_to_id, id_to_endpoint};

// It is very important to block on any request made via the service, since ordering is
// crucial when registering operations or creating tensors. The `DeviceHandle` queue
// preserves submission order, so `submit` is sufficient for cheap fire-and-forget ops; we
// only `submit_blocking` for paths that need to read the service's response.
impl<C: ProtocolClient> RouterClient for RemoteClient<C> {
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
        let shape = data.shape.clone();
        let dtype = data.dtype;
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

impl<C: ProtocolClient> RemoteClient<C> {
    /// Rewrite the device ids carried by an op so the server can resolve them.
    ///
    /// This runs for every op, but only ops that carry device ids (currently the collective ops)
    /// are affected On the client, the participating devices are identified by their *remote*
    /// device ids (`type_id = 0`, `index_id = ` the local-registry index that encodes
    /// `address`+device index). The server can't reverse that registry hash, so we translate each
    /// id to the plain server-local device index (kept in `index_id`, `type_id` left 0). The
    /// server then maps each index to its own backend device id before executing — see
    /// `RemoteServer::resolve_devices`.
    ///
    /// Only same-server collectives are supported for now: every participating device must live
    /// on the same address as the tensor's device. A cross-server group panics with a clear
    /// message rather than silently reducing the wrong devices.
    fn resolve_devices(&self, mut op: burn_ir::OperationIr) -> burn_ir::OperationIr {
        use burn_ir::{DistributedOperationIr, OperationIr};

        if let OperationIr::Distributed(DistributedOperationIr::AllReduce(desc)) = &mut op {
            let local_address = self.device.address();
            for id in desc.device_ids.iter_mut() {
                let (address, device_index) = id_to_endpoint(id.index_id as u32).expect(
                    "an all_reduce device must be a registered remote device on this process",
                );
                assert_eq!(
                    address, local_address,
                    "cross-server all_reduce is not supported yet: the tensor is on `{local_address}` \
                     but the collective includes a device on `{address}`",
                );
                id.type_id = 0;
                id.index_id = device_index as u16;
            }
            log::trace!(
                "All-reduce on {:?} ({local_address}): {desc:?}",
                self.device
            );
        }

        op
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
/// The device contains the connection information of the server plus the index of the device
/// to select on it.
///
/// Two `RemoteDevice`s that share an `address` but differ in `device_index` point at distinct
/// devices on the same server; they get distinct registry ids (and thus distinct service
/// threads/connections), and a transfer between them is detected as same-host.
pub struct RemoteDevice {
    pub(crate) address: Address,
    /// The index of the device to select on the server (see [`endpoint_to_id`]).
    pub(crate) device_index: u32,
    /// The id of the device in the local registry, see [`endpoint_to_id`].
    pub(crate) id: u32,
}

impl RemoteDevice {
    /// Create a device from a url and the index of the device to select on the server.
    pub fn new(address: &str, device_index: usize) -> Self {
        let address = Address::from_str(address).expect("Could not parse remote address");
        let device_index = device_index as u32;
        // Key the registry on the canonical address so equivalent spellings share one id.
        let id = endpoint_to_id(address.to_string(), device_index);
        Self {
            address,
            device_index,
            id,
        }
    }

    /// Forces the client connection to be established immediately using the default protocol.
    /// This is a no-op if the connection is already up for this device.
    pub fn connect(&self) {
        use burn_communication::Protocol;
        type DefaultChannel = RemoteChannel<<crate::shared::RemoteProtocol as Protocol>::Client>;

        // `get_client` initializes the (lazy) service if needed; `ensure_connected` then opens
        // the sockets and runs the handshake on the runner thread, so the settings/device-count
        // cells are populated by the time we return.
        get_client::<DefaultChannel>(self).ensure_connected();
    }

    /// Initializes the client for this device using the specified protocol channel.
    /// This is a no-op if the client already exists for this address.
    ///
    /// Note this only creates the (lazy) service — the actual socket connection and handshake
    /// open on first use. Use [`connect`](Self::connect) when you need the connection (and the
    /// device's settings) established right away.
    pub fn connect_with_channel<R: burn_router::RouterChannel<Device = Self>>(&self) {
        // `get_client` forces service initialization if the client doesn't exist yet;
        // `RemoteService::init` records the endpoint but defers the connect to first use.
        get_client::<R>(self);
    }

    /// The canonical network address of the server this device lives on.
    pub fn address(&self) -> String {
        self.address.to_string()
    }

    /// The index of this device on its server.
    pub fn device_index(&self) -> usize {
        self.device_index as usize
    }

    /// List every device hosted by the server at `address`, one [`RemoteDevice`] per device
    /// index the server exposes.
    ///
    /// Connecting is required to learn how many devices the server hosts (the count rides on
    /// the init handshake), so this establishes the connection to the server's default device
    /// (index 0). The returned devices for the remaining indices connect lazily on first use,
    /// matching [`Device::enumerate`](burn_backend::tensor::Device)'s behavior for local
    /// backends.
    pub fn enumerate(address: &str) -> Vec<Self> {
        // Device 0 always exists (a server must host at least one device); connecting to it
        // populates the device-count cell for its registry id.
        let device = Self::new(address, 0);
        device.connect();

        let count = service::device_count_for(device.id)
            .expect("Device count populated by the init handshake during connect");

        (0..count as usize)
            .map(|index| Self::new(address, index))
            .collect()
    }
}

impl Default for RemoteDevice {
    fn default() -> Self {
        let address = match std::env::var("BURN_REMOTE_ADDRESS") {
            Ok(address) => address,
            Err(_) => String::from("ws://127.0.0.1:3000"),
        };

        Self::new(&address, 0)
    }
}

impl burn_std::device::Device for RemoteDevice {
    fn from_id(device_id: DeviceId) -> Self {
        if device_id.type_id != 0 {
            panic!("Invalid device id: {device_id} (expected type 0)");
        }
        let (address, device_index) = id_to_endpoint(device_id.index_id as u32)
            .unwrap_or_else(|| panic!("Invalid device id: {device_id}"));
        Self::new(&address, device_index as usize)
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
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
        if !service::has_settings(self.id) {
            self.connect();
        }
        service::settings_for(self.id)
    }
}

pub struct RemoteBridge<C: ProtocolClient> {
    _p: PhantomData<C>,
}

pub struct RemoteTensorHandle<C: ProtocolClient> {
    pub(crate) client: RemoteClient<C>,
    pub(crate) tensor: TensorIr,
    pub(crate) _p: PhantomData<C>,
}

static TRANSFER_COUNTER: Mutex<Option<TensorTransferId>> = Mutex::new(None);

/// Allocate the next globally-unique [`TensorTransferId`] for a same-host / cross-server transfer.
///
/// The id keys the server's transfer rendezvous (`local_comm` / `external_comm`), so two
/// transfers that are ever in flight at the same time MUST get distinct ids — otherwise a `take`
/// can pick up the wrong (or an overwritten) exposed primitive and its peer hangs forever. The
/// counter is incremented **in place** in the static; `TensorTransferId` is `Copy`, so the
/// earlier `transfer_counter.unwrap()` copied the value out and incremented a throwaway local,
/// leaving every transfer after the first sharing id 1 — harmless sequentially, a deadlock under
/// concurrency.
fn get_next_transfer_id() -> TensorTransferId {
    let mut transfer_counter = TRANSFER_COUNTER.lock().unwrap();
    match transfer_counter.as_mut() {
        Some(id) => {
            id.next();
            *id
        }
        None => {
            let id = TensorTransferId::from(0);
            *transfer_counter = Some(id);
            id
        }
    }
}

impl<C: ProtocolClient> RemoteTensorHandle<C> {
    /// Move the tensor to `target_device`, picking the cheapest path.
    ///
    /// When the source and target live on the **same** server (same address, different device
    /// index), the data never leaves the process — see [`change_backend_local`]. Otherwise we
    /// fall back to the cross-server path that streams the data server-to-server without the
    /// client ever seeing it.
    pub(crate) fn change_backend(self, target_device: &RemoteDevice) -> Self {
        if self.client.device.address == target_device.address {
            self.change_backend_local(target_device)
        } else {
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

        let target_client = get_client::<RemoteChannel<C>>(target_device);
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
        let transfer_id = get_next_transfer_id();
        let tensor = self.tensor.clone();
        self.client.handle.submit(move |s| {
            s.expose_tensor_remote(stream_id, tensor, 1, transfer_id);
        });
        // `submit` only enqueues the closure on the device-runner queue; the runner
        // wouldn't drain it until 32 ops accumulated. `flush_queue` forces the runner
        // thread to run the closure now, and `expose_tensor_remote` itself flushes the
        // service batch onto the wire — so the source server receives the expose before
        // the target server starts trying to download.
        self.client.handle.flush_queue();

        let target_client = get_client::<RemoteChannel<C>>(target_device);

        let address = self.client.device.address.clone();
        let new_id = service::new_tensor_id();
        target_client.handle.submit(move |s| {
            s.register_tensor_remote(
                stream_id,
                TensorRemote {
                    transfer_id,
                    address,
                },
                new_id,
            );
        });
        // Same as the source side: drain the closure queue so it runs now and
        // `register_tensor_remote` flushes the registration onto the target's wire.
        target_client.handle.flush_queue();

        self.tensor.id = new_id;
        self.client = target_client;

        self
    }
}

impl<C: ProtocolClient> MultiBackendBridge for RemoteBridge<C> {
    type TensorHandle = RemoteTensorHandle<C>;
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
