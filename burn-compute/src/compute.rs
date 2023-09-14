use hashbrown::HashMap;

use crate::{ComputeChannel, ComputeClient, ComputeServer, MutexComputeChannel};

/// The compute type has the responsability to retrive the correct compute client based on the
/// given device.
///
/// This type can be used as a singleton for your backend implementation.
///
/// # Example
///
/// You can wrap the compute struct into a mutex to access a compute client from a single function:
///
/// ```rust, ignore
/// static COMPUTE: Mutex<Compute<Device, Server, Channel>> = Mutex::new(Compute::new());
///
/// pub fn get(device: &Device) -> ComputeClient<Server, Channel> {
///    let mut compute = COMPUTE.lock();
///
///    compute.get(device, || {
///        let storage = BytesStorage::default();
///        let memory_management = SimpleMemoryManagement::never_dealloc(storage);
///        let server = Server::new(memory_management);
///        let channel = Channel::new(server);
///
///        ComputeClient::new(channel)
///    })
/// }
///
/// ```
/// Note that you can clone the clients without problem, so the static `get` function is only useful
/// when retriving a client with only the device information.
pub struct Compute<Device, Server, Channel = MutexComputeChannel<Server>> {
    clients: Option<HashMap<Device, ComputeClient<Server, Channel>>>,
}

impl<Device, Server, Channel> Compute<Device, Server, Channel>
where
    Device: core::hash::Hash + PartialEq + Eq + Clone + core::fmt::Debug,
    Server: ComputeServer,
    Channel: ComputeChannel<Server>,
{
    /// Create a new compute.
    pub const fn new() -> Self {
        Self { clients: None }
    }

    /// Get the compute client for the given device.
    ///
    /// Provide the init function to create a new client if it isn't already initialized.
    pub fn get<Init>(&mut self, device: &Device, init: Init) -> ComputeClient<Server, Channel>
    where
        Init: Fn() -> ComputeClient<Server, Channel>,
    {
        let clients = self.clients();

        if let Some(client) = clients.get(device) {
            client.clone()
        } else {
            let client = init();
            clients.insert(device.clone(), client.clone());
            client
        }
    }

    /// Register the compute client for the given device.
    ///
    /// # Note
    ///
    /// This function is mostly useful when the creation of the compute client can't be done
    /// syunchonously and require special contexte.
    ///
    /// # Panics
    ///
    /// If a client is already registered for the given device.
    pub fn register(&mut self, device: &Device, client: ComputeClient<Server, Channel>) {
        let clients = self.clients();

        if clients.contains_key(device) {
            panic!("Client already created for device {:?}", device);
        }

        clients.insert(device.clone(), client.clone());
    }

    fn clients<'a>(&'a mut self) -> &'a mut HashMap<Device, ComputeClient<Server, Channel>> {
        self.init();

        if let Some(clients) = &mut self.clients {
            return clients;
        }

        unreachable!();
    }

    fn init(&mut self) {
        if let None = self.clients {
            self.clients = Some(HashMap::new());
        }
    }
}
