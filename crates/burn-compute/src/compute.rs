use crate::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};
use core::ops::DerefMut;
use hashbrown::HashMap;

/// The compute type has the responsibility to retrieve the correct compute client based on the
/// given device.
pub struct ComputeRuntime<Device, Server: ComputeServer, Channel> {
    clients: spin::Mutex<Option<HashMap<Device, ComputeClient<Server, Channel>>>>,
}

impl<Device, Server, Channel> Default for ComputeRuntime<Device, Server, Channel>
where
    Device: core::hash::Hash + PartialEq + Eq + Clone + core::fmt::Debug,
    Server: ComputeServer,
    Channel: ComputeChannel<Server>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Device, Server, Channel> ComputeRuntime<Device, Server, Channel>
where
    Device: core::hash::Hash + PartialEq + Eq + Clone + core::fmt::Debug,
    Server: ComputeServer,
    Channel: ComputeChannel<Server>,
{
    /// Create a new compute.
    pub const fn new() -> Self {
        Self {
            clients: spin::Mutex::new(None),
        }
    }

    /// Get the compute client for the given device.
    ///
    /// Provide the init function to create a new client if it isn't already initialized.
    pub fn client<Init>(&self, device: &Device, init: Init) -> ComputeClient<Server, Channel>
    where
        Init: Fn() -> ComputeClient<Server, Channel>,
    {
        let mut clients = self.clients.lock();

        if clients.is_none() {
            Self::register_inner(device, init(), &mut clients);
        }

        match clients.deref_mut() {
            Some(clients) => match clients.get(device) {
                Some(client) => client.clone(),
                None => {
                    let client = init();
                    clients.insert(device.clone(), client.clone());
                    client
                }
            },
            _ => unreachable!(),
        }
    }

    /// Register the compute client for the given device.
    ///
    /// # Note
    ///
    /// This function is mostly useful when the creation of the compute client can't be done
    /// synchronously and require special context.
    ///
    /// # Panics
    ///
    /// If a client is already registered for the given device.
    pub fn register(&self, device: &Device, client: ComputeClient<Server, Channel>) {
        let mut clients = self.clients.lock();

        Self::register_inner(device, client, &mut clients);
    }

    fn register_inner(
        device: &Device,
        client: ComputeClient<Server, Channel>,
        clients: &mut Option<HashMap<Device, ComputeClient<Server, Channel>>>,
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
