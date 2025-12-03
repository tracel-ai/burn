use super::{RemoteChannel, RemoteClient};
use crate::shared::{ComputeTask, TaskResponseContent, TensorRemote};
use burn_communication::{Address, ProtocolClient, data_service::TensorTransferId};
use burn_ir::TensorIr;
use burn_router::{MultiBackendBridge, RouterTensor, RunnerClient, get_client};
use burn_std::future::DynFut;
use burn_tensor::{
    Shape, TensorData,
    backend::{DeviceId, DeviceOps, ExecutionError},
};
use std::sync::OnceLock;
use std::{collections::HashMap, marker::PhantomData, str::FromStr, sync::Mutex};

// TODO: we should work with the parsed structure of Address, not the string.
static ADDRESS_REGISTRY: OnceLock<Mutex<HashMap<String, u32>>> = OnceLock::new();

fn get_address_registry() -> &'static Mutex<HashMap<String, u32>> {
    ADDRESS_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Map a string network address to a (local runtime) global unique u32.
///
/// Globally stable over the lifetime of the process, shared between threads,
/// If the address has never been seen, a new id will be created.
/// If the address has been seen, the previous id will be returned.
pub fn address_to_id<S: AsRef<str>>(address: S) -> u32 {
    let registry = get_address_registry();
    let mut registry = registry.lock().unwrap();
    let next_id = registry.len() as u32;
    *registry
        .entry(address.as_ref().to_string())
        .or_insert_with(|| next_id)
}

/// Look up an address by id.
///
/// Returns the same address given ids by [`address_to_id`].
pub fn id_to_address(id: u32) -> Option<String> {
    let registry = get_address_registry();
    let registry = registry.lock().unwrap();
    for entry in registry.iter() {
        if entry.1 == &id {
            return Some(entry.0.clone());
        }
    }
    None
}

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

    fn read_tensor_async(
        &self,
        tensor: burn_ir::TensorIr,
    ) -> DynFut<Result<TensorData, ExecutionError>> {
        // Important for ordering to call the creation of the future sync.
        let fut = self.sender.send_async(ComputeTask::ReadTensor(tensor));

        Box::pin(async move {
            match fut.await {
                Ok(response) => match response {
                    TaskResponseContent::ReadTensor(res) => res,
                    _ => panic!("Invalid message type"),
                },
                Err(e) => Err(ExecutionError::Generic {
                    context: format!("Failed to read tensor: {:?}", e),
                }),
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

    fn sync(&self) -> Result<(), ExecutionError> {
        // Important for ordering to call the creation of the future sync.
        let fut = self.sender.send_async(ComputeTask::SyncBackend);

        match self.runtime.block_on(fut) {
            Ok(response) => match response {
                TaskResponseContent::SyncBackend(res) => res,
                _ => panic!("Invalid message type"),
            },
            Err(e) => Err(SyncError::Generic {
                context: format!("Failed to sync: {:?}", e),
            }),
        }
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
    /// The id of the device in the local registry, see [`address_to_id`].
    pub(crate) id: u32,
}

impl RemoteDevice {
    /// Create a device from an url.
    pub fn new(address: &str) -> Self {
        let id = address_to_id(address);
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

impl burn_std::device::Device for RemoteDevice {
    fn from_id(device_id: DeviceId) -> Self {
        if device_id.type_id != 0 {
            panic!("Invalid device id: {device_id} (expected type 0)");
        }
        let address = id_to_address(device_id.index_id)
            .unwrap_or_else(|| panic!("Invalid device id: {device_id}"));
        Self::new(&address)
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
