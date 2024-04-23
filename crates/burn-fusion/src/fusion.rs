use burn_tensor::{
    backend::{Backend, DeviceId, DeviceOps},
    repr::ReprBackend,
};

use crate::client::FusionClient;

use std::{any::Any, collections::HashMap, ops::DerefMut};

/// Type alias for [representation backend handle](burn_tensor::repr::ReprBackend::Handle).
pub type Handle<B> = <B as ReprBackend>::Handle;
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
    pub fn client<C: FusionClient + 'static>(
        &self,
        device: &<C::FusionBackend as Backend>::Device,
    ) -> C {
        let device_id = device.id();
        let client_id = (core::any::TypeId::of::<C>(), device_id);
        let mut clients = self.clients.lock();

        if clients.is_none() {
            let client = C::new(device.clone());
            Self::register_inner::<C>(client_id, client, &mut clients);
        }

        match clients.deref_mut() {
            Some(clients) => match clients.get(&client_id) {
                Some(client) => {
                    let client: &C = client.downcast_ref().unwrap();
                    client.clone()
                }
                None => {
                    let client = C::new(device.clone());
                    let any = Box::new(client.clone());
                    clients.insert(client_id, any);
                    client
                }
            },
            _ => unreachable!(),
        }
    }

    fn register_inner<C: FusionClient + 'static>(
        key: Key,
        client: C,
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
