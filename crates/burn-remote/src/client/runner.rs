use super::{RemoteChannel, RemoteClient, service};
use crate::shared::{TaskResponseContent, TensorRemote};
use burn_backend::{DeviceId, DeviceOps, ExecutionError, StreamId, TensorData};
use burn_communication::{Address, ProtocolClient, data_service::TensorTransferId};
use burn_ir::TensorIr;
use burn_router::{MultiBackendBridge, RouterTensor, RunnerClient, get_client};
use burn_std::DeviceSettings;
use burn_std::{backtrace::BackTrace, future::DynFut};
use std::sync::Mutex;
use std::{marker::PhantomData, str::FromStr};

pub use service::{address_to_id, id_to_address};

// It is very important to block on any request made via the service, since ordering is
// crucial when registering operations or creating tensors. The `DeviceHandle` queue
// preserves submission order, so `submit` is sufficient for cheap fire-and-forget ops; we
// only `submit_blocking` for paths that need to read the service's response.
impl<C: ProtocolClient> RunnerClient for RemoteClient<C> {
    type Device = RemoteDevice;

    fn register_op(&self, op: burn_ir::OperationIr) {
        let stream_id = StreamId::current();
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

    fn dtype_usage(&self, dtype: burn_std::DType) -> burn_backend::DTypeUsageSet {
        self.handle
            .submit_blocking(move |s| s.dtype_usage(dtype))
            .expect("Service call failed")
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
/// The device contains the connection information of the server.
pub struct RemoteDevice {
    pub(crate) address: Address,
    /// The id of the device in the local registry, see [`address_to_id`].
    pub(crate) id: u32,
}

impl RemoteDevice {
    /// Create a device from a url.
    pub fn new(address: &str) -> Self {
        let id = address_to_id(address);
        Self {
            address: Address::from_str(address).unwrap(),
            id,
        }
    }

    /// Forces the client connection to be established immediately using the default protocol.
    /// This is a no-op if the client is already initialized for this address.
    pub fn connect(&self) {
        use burn_communication::Protocol;
        type DefaultChannel = RemoteChannel<<crate::shared::RemoteProtocol as Protocol>::Client>;

        self.connect_with_channel::<DefaultChannel>();
    }

    /// Forces the connection using the specified communication protocol channel.
    /// This is a no-op if the client is already initialized for this address.
    pub fn connect_with_channel<R: burn_router::RunnerChannel<Device = Self>>(&self) {
        // If the client doesn't exist yet, `get_client` forces initialization, which in
        // turn calls `RemoteService::init` and populates the settings for this device.
        get_client::<R>(self);
    }
}

impl Default for RemoteDevice {
    fn default() -> Self {
        let address = match std::env::var("BURN_REMOTE_ADDRESS") {
            Ok(address) => address,
            Err(_) => String::from("ws://127.0.0.1:3000"),
        };

        Self::new(&address)
    }
}

impl burn_std::device::Device for RemoteDevice {
    fn from_id(device_id: DeviceId) -> Self {
        if device_id.type_id != 0 {
            panic!("Invalid device id: {device_id} (expected type 0)");
        }
        let address = id_to_address(device_id.index_id as u32)
            .unwrap_or_else(|| panic!("Invalid device id: {device_id}"));
        Self::new(&address)
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

fn get_next_transfer_id() -> TensorTransferId {
    let mut transfer_counter = TRANSFER_COUNTER.lock().unwrap();
    if transfer_counter.is_none() {
        *transfer_counter = Some(0.into());

        transfer_counter.unwrap()
    } else {
        let mut transfer_counter = transfer_counter.unwrap();
        transfer_counter.next();

        transfer_counter
    }
}

impl<C: ProtocolClient> RemoteTensorHandle<C> {
    /// Changes the backend of the tensor via a dWebSocket.
    /// We ask the original server to expose the tensor, then ask the target server to fetch
    /// the tensor. The target server will open a new network connection to the original server
    /// to download the data.
    /// This way the client never sees the tensor's data, and we avoid a bottleneck.
    pub(crate) fn change_backend(mut self, target_device: &RemoteDevice) -> Self {
        let transfer_id = get_next_transfer_id();
        let tensor = self.tensor.clone();
        self.client.handle.submit(move |s| {
            s.expose_tensor_remote(tensor, 1, transfer_id);
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
