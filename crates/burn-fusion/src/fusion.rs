use burn_ir::BackendIr;
use burn_tensor::backend::{DeviceId, DeviceOps};

use crate::{client::FusionClient, Client, FusionDevice, FusionRuntime};

use std::{any::Any, collections::HashMap, ops::DerefMut};

/// Type alias for [representation backend handle](burn_ir::BackendIr::Handle).
pub type Handle<B> = <B as BackendIr>::Handle;
type Key = (core::any::TypeId, DeviceId);

pub(crate) struct FusionClientLocator {
    clients: spin::Mutex<Option<HashMap<Key, Box<dyn core::any::Any + Send>>>>,
}

impl FusionClientLocator {
    /// Create a new client locator.
    pub const fn new() -> Self {
        Self {
            clients: spin::Mutex::new(None),
        }
    }

    /// Get the fusion client for the given device.
    ///
    /// Provide the init function to create a new client if it isn't already initialized.
    pub fn client<R: FusionRuntime + 'static>(&self, device: &FusionDevice<R>) -> Client<R> {
        let device_id = device.id();
        let client_id = (core::any::TypeId::of::<R>(), device_id);
        let mut clients = self.clients.lock();

        if clients.is_none() {
            let client = Client::<R>::new(device.clone());
            Self::register_inner::<R>(client_id, client, &mut clients);
        }

        match clients.deref_mut() {
            Some(clients) => match clients.get(&client_id) {
                Some(client) => {
                    let client: &Client<R> = client.downcast_ref().unwrap();
                    client.clone()
                }
                None => {
                    let client = Client::<R>::new(device.clone());
                    let any = Box::new(client.clone());
                    clients.insert(client_id, any);
                    client
                }
            },
            _ => unreachable!(),
        }
    }

    fn register_inner<R: FusionRuntime + 'static>(
        key: Key,
        client: Client<R>,
        clients: &mut Option<HashMap<Key, Box<dyn Any + Send>>>,
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
