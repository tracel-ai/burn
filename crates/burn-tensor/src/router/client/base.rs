use core::{future::Future, ops::DerefMut};
use std::collections::HashMap;

use burn_common::stream::StreamId;
use spin::Mutex;

use crate::{
    backend::{DeviceId, DeviceOps},
    repr::{OperationDescription, TensorDescription},
    router::RunnerChannel,
    DType, TensorData,
};

/// Type alias for `<R as RunnerChannel>::Client`.
pub type Client<R> = <R as RunnerChannel>::Client;
pub(crate) static CLIENTS: RunnerClientLocator = RunnerClientLocator::new();
type Key = (core::any::TypeId, DeviceId);

/// Define how to interact with the runner server.
pub trait RunnerClient: Clone + Send + Sync + Sized {
    /// Register a new tensor operation to be executed by the (runner) server.
    fn register(&self, op: OperationDescription, stream: StreamId);
    /// Read the values contained by a tensor.
    fn read_tensor(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send;
    fn write_tensor(&self, data: TensorData, stream: StreamId) -> TensorDescription;
    fn empty_tensor(&self, shape: Vec<usize>, dtype: DType, stream: StreamId) -> TensorDescription;
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
