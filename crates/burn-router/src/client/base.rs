use alloc::{boxed::Box, vec::Vec};
use core::{future::Future, ops::DerefMut};
use hashbrown::HashMap;

use spin::Mutex;

use burn_tensor::{
    backend::{DeviceId, DeviceOps, SyncType},
    repr::{OperationDescription, TensorDescription, TensorId},
    DType, TensorData,
};

use crate::{RouterTensor, RunnerChannel};

/// Type alias for `<R as RunnerChannel>::Client`.
pub type Client<R> = <R as RunnerChannel>::Client;
pub(crate) static CLIENTS: RunnerClientLocator = RunnerClientLocator::new();
type Key = (core::any::TypeId, DeviceId);

/// Define how to interact with the runner.
pub trait RunnerClient: Clone + Send + Sync + Sized {
    /// Device type.
    type Device: DeviceOps;

    /// Register a new tensor operation to be executed by the (runner) server.
    fn register(&self, op: OperationDescription);
    /// Read the values contained by a tensor.
    fn read_tensor(&self, tensor: TensorDescription) -> impl Future<Output = TensorData> + Send;
    /// Create a new [RouterTensor] from the tensor data.
    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self>;
    /// Create a new [RouterTensor] with no resources associated.
    fn register_empty_tensor(&self, shape: Vec<usize>, dtype: DType) -> RouterTensor<Self>;
    /// Create a new float [RouterTensor] with no resources associated.
    fn register_float_tensor(&self, shape: Vec<usize>, full_precision: bool) -> RouterTensor<Self>;
    /// Get the current device used by all operations handled by this client.
    fn device(&self) -> Self::Device;
    /// Drop the tensor with the given [tensor id](TensorId).
    fn register_orphan(&self, id: &TensorId);
    /// Sync the runner, ensure that all computations are finished.
    fn sync(&self, sync_type: SyncType);
}

pub(crate) struct RunnerClientLocator {
    clients: Mutex<Option<HashMap<Key, Box<dyn core::any::Any + Send>>>>,
}

pub(crate) fn get_client<R: RunnerChannel>(device: &R::Device) -> Client<R> {
    CLIENTS.client::<R>(device)
}

impl RunnerClientLocator {
    /// Create a new client locator.
    pub const fn new() -> Self {
        Self {
            clients: Mutex::new(None),
        }
    }

    /// Get the runner client for the given device.
    ///
    /// If a client isn't already initialized, it is created.
    pub fn client<R: RunnerChannel + 'static>(&self, device: &R::Device) -> Client<R> {
        let device_id = device.id();
        let client_id = (core::any::TypeId::of::<R>(), device_id);
        let mut clients = self.clients.lock();

        if clients.is_none() {
            let client = R::init_client(device);
            Self::register_inner::<R>(client_id, client, &mut clients);
        }

        match clients.deref_mut() {
            Some(clients) => match clients.get(&client_id) {
                Some(client) => {
                    let client: &Client<R> = client.downcast_ref().unwrap();
                    client.clone()
                }
                None => {
                    let client = R::init_client(device);
                    let any = Box::new(client.clone());
                    clients.insert(client_id, any);
                    client
                }
            },
            _ => unreachable!(),
        }
    }

    fn register_inner<R: RunnerChannel + 'static>(
        key: Key,
        client: Client<R>,
        clients: &mut Option<HashMap<Key, Box<dyn core::any::Any + Send>>>,
    ) {
        if clients.is_none() {
            *clients = Some(HashMap::new());
        }

        if let Some(clients) = clients {
            if clients.contains_key(&key) {
                panic!("Client already created for device {:?}", key);
            }

            clients.insert(key, Box::new(client));
        }
    }
}
