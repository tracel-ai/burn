use burn_common::future::DynFut;
use burn_communication::{Address, ProtocolClient, data_service::TensorTransferId};
use burn_ir::TensorIr;
use burn_router::{MultiBackendBridge, RouterTensor, RunnerClient, get_client};
use burn_tensor::{
    Shape, TensorData,
    backend::{DeviceId, DeviceOps},
};
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    str::FromStr,
    sync::Mutex,
};

use crate::shared::{ComputeTask, TaskResponseContent, TensorRemote};

use super::{RemoteChannel, RemoteClient};

// It is very important to block on any request made with the sender, since ordering is crucial
// when registering operation or creating tensors.
//
// The overhead is minimal, since we only wait for the task to be sent to the async
// channel, but not sent to the server and even less processed by the server.
impl RunnerClient for RemoteClient {
    type Device = RemoteDevice;

    fn register_op(&self, op: burn_ir::OperationIr) {
        self.sender
            .send(ComputeTask::RegisterOperation(Box::new(op)));
    }

    fn read_tensor(&self, tensor: burn_ir::TensorIr) -> DynFut<TensorData> {
        // Important for ordering to call the creation of the future sync.
        let fut = self.sender.send_callback(ComputeTask::ReadTensor(tensor));

        Box::pin(async move {
            match fut.await {
                TaskResponseContent::ReadTensor(data) => data,
                _ => panic!("Invalid message type"),
            }
        })
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let id = self.sender.new_tensor_id();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        self.sender.send(ComputeTask::RegisterTensor(id, data));

        RouterTensor::new(id, Shape::from(shape), dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn sync(&self) {
        // Important for ordering to call the creation of the future sync.
        let fut = self.sender.send_callback(ComputeTask::SyncBackend);

        let runtime = self.runtime.clone();

        match runtime.block_on(fut) {
            TaskResponseContent::SyncBackend => {}
            _ => panic!("Invalid message type"),
        };
    }

    fn seed(&self, seed: u64) {
        self.sender.send(ComputeTask::Seed(seed));
    }

    fn create_empty_handle(&self) -> burn_ir::TensorId {
        self.sender.new_tensor_id()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
/// The device contains the connection information of the server.
pub struct RemoteDevice {
    pub(crate) address: Address,
    // Unique ID generated from hash of the address
    pub(crate) id: u32,
}

impl RemoteDevice {
    /// Create a device from an url.
    pub fn new(address: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        address.hash(&mut hasher);
        let id = hasher.finish() as u32;

        Self {
            address: Address::from_str(address).unwrap(),
            id,
        }
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

impl burn_common::device::Device for RemoteDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        todo!("Should keep the address as ints, host should be type, port should be index.")
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.id,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

impl DeviceOps for RemoteDevice {}

pub struct RemoteBridge<C: ProtocolClient> {
    _p: PhantomData<C>,
}

pub struct RemoteTensorHandle<C: ProtocolClient> {
    pub(crate) client: RemoteClient,
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
        self.client.sender.send(ComputeTask::ExposeTensorRemote {
            tensor: self.tensor.clone(),
            count: 1,
            transfer_id,
        });

        let target_client = get_client::<RemoteChannel<C>>(target_device);

        let new_id = target_client.sender.new_tensor_id();

        let remote_tensor = TensorRemote {
            transfer_id,
            address: self.client.device.address.clone(),
        };
        target_client
            .sender
            .send(ComputeTask::RegisterTensorRemote(remote_tensor, new_id));

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
        _shape: burn_tensor::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }

    fn change_backend_int(
        tensor: Self::TensorHandle,
        _shape: burn_tensor::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }

    fn change_backend_bool(
        tensor: Self::TensorHandle,
        _shape: burn_tensor::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }
}
