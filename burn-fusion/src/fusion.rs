use crate::{client::FusionClient, FusedBackend, FusionServer};
use burn_tensor::backend::Backend;
use std::{collections::HashMap, ops::DerefMut};

pub type Handle<B> = <B as FusedBackend>::Handle;
pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type IntElem<B> = <B as Backend>::IntElem;

pub struct FusionClientLocator<C: FusionClient> {
    clients: spin::Mutex<Option<HashMap<<C::FusedBackend as FusedBackend>::HandleDevice, C>>>,
}

impl<C: FusionClient> FusionClientLocator<C> {
    /// Create a new compute.
    pub const fn new() -> Self {
        Self {
            clients: spin::Mutex::new(None),
        }
    }

    /// Get the fusion client for the given device.
    ///
    /// Provide the init function to create a new client if it isn't already initialized.
    pub fn client(&self, device: &<C::FusedBackend as FusedBackend>::HandleDevice) -> C {
        let mut clients = self.clients.lock();

        if clients.is_none() {
            let client = C::new(FusionServer::new(device.clone()));
            Self::register_inner(device, client, &mut clients);
        }

        match clients.deref_mut() {
            Some(clients) => match clients.get(device) {
                Some(client) => client.clone(),
                None => {
                    let client = C::new(FusionServer::new(device.clone()));
                    clients.insert(device.clone(), client.clone());
                    client
                }
            },
            _ => unreachable!(),
        }
    }

    fn register_inner(
        device: &<C::FusedBackend as FusedBackend>::HandleDevice,
        client: C,
        clients: &mut Option<HashMap<<C::FusedBackend as FusedBackend>::HandleDevice, C>>,
    ) {
        if clients.is_none() {
            *clients = Some(HashMap::new());
        }

        if let Some(clients) = clients {
            if clients.contains_key(device) {
                panic!("Client already created for device {:?}", device);
            }

            clients.insert(device.clone(), client);
        }
    }
}
