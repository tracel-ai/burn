use burn_ir::BackendIr;
use burn_tensor::backend::{DeviceId, DeviceOps};

use crate::{Client, FusionDevice, FusionRuntime, client::FusionClient};

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    ops::DerefMut,
};
use spin::Mutex;

/// Type alias for [representation backend handle](burn_ir::BackendIr::Handle).
pub type Handle<B> = <B as BackendIr>::Handle;

/// Key = (runtime tipi, device id)
type Key = (TypeId, DeviceId);

pub(crate) struct FusionClientLocator {
    clients: Mutex<HashMap<Key, Box<dyn Any + Send>>>,
}

impl FusionClientLocator {
    /// Create a new client locator.
    pub fn new() -> Self {
        Self {
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// Get the fusion client for the given device.
    ///
    /// If it doesn't exist, lazily initializes it.
    pub fn client<R: FusionRuntime + 'static>(&self, device: &FusionDevice<R>) -> Client<R> {
        let device_id = device.id();
        let client_id = (TypeId::of::<R>(), device_id);

        let mut clients = self.clients.lock();

        // Already exists?
        if let Some(existing) = clients.get(&client_id) {
            existing
                .downcast_ref::<Client<R>>()
                .expect("Client type mismatch.")
                .clone()
        } else {
            // Otherwise, create and insert
            let client = Client::<R>::new(device.clone());
            clients.insert(client_id, Box::new(client.clone()));
            client
        }
    }
}
