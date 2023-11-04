use burn_tensor::backend::Backend;

use crate::{
    channel::{mutex::MutexFusionChannel, FusionChannel},
    graph::{FusedBackend, GraphExecution, GreedyGraphExecution},
    FusionClient, FusionServer,
};
use std::{collections::HashMap, ops::DerefMut};

pub type Client<B> =
    FusionClient<B, MutexFusionChannel<B, GreedyGraphExecution>, GreedyGraphExecution>;

pub type Handle<B> = <B as FusedBackend>::Handle;
pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type IntElem<B> = <B as Backend>::IntElem;

pub struct Fusion<B: FusedBackend, C, G> {
    clients: spin::Mutex<Option<HashMap<B::HandleDevice, FusionClient<B, C, G>>>>,
}

impl<B, C, G> Fusion<B, C, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
    C: FusionChannel<B, G>,
{
    /// Create a new compute.
    pub const fn new() -> Self {
        Self {
            clients: spin::Mutex::new(None),
        }
    }

    /// Get the fusion client for the given device.
    ///
    /// Provide the init function to create a new client if it isn't already initialized.
    pub fn client(&self, device: &B::HandleDevice) -> FusionClient<B, C, G> {
        let mut clients = self.clients.lock();

        if clients.is_none() {
            let client = FusionClient::new(C::new(FusionServer::new(device.clone())));
            Self::register_inner(device, client, &mut clients);
        }

        match clients.deref_mut() {
            Some(clients) => match clients.get(device) {
                Some(client) => client.clone(),
                None => {
                    let client = FusionClient::new(C::new(FusionServer::new(device.clone())));
                    clients.insert(device.clone(), client.clone());
                    client
                }
            },
            _ => unreachable!(),
        }
    }

    fn register_inner(
        device: &B::HandleDevice,
        client: FusionClient<B, C, G>,
        clients: &mut Option<HashMap<B::HandleDevice, FusionClient<B, C, G>>>,
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
