use std::{
    any::{Any, TypeId},
    sync::{Mutex, MutexGuard, OnceLock},
};

use crate::{collections::HashMap, grad_sync::client::GradientSyncClient};
use burn_backend::Backend;

/// Errors from gradient syncing operations
#[allow(unused)]
#[derive(Debug, Clone)]
pub enum GradientSyncError {
    #[allow(unused)]
    Other(String),
}

/// The type-erased box type for [`GradientSyncClient`].
type ClientBox = Box<dyn Any + Send + Sync>;

/// Global state map from [`Backend`] to boxed [`GradientSyncClient`].
static BACKEND_CLIENT_MAP: OnceLock<Mutex<HashMap<TypeId, ClientBox>>> = OnceLock::new();

/// Gets a locked mutable view of the `STATE_MAP`.
pub(crate) fn get_backend_client_map() -> MutexGuard<'static, HashMap<TypeId, ClientBox>> {
    BACKEND_CLIENT_MAP
        .get_or_init(Default::default)
        .lock()
        .unwrap()
}

/// Get a [`GradientSyncClient`] for the given [`Backend`].
pub(crate) fn get_gradient_sync_client<B: Backend>() -> Option<GradientSyncClient<B>> {
    let typeid = TypeId::of::<B>();
    let state_map = get_backend_client_map();
    match state_map.get(&typeid) {
        Some(val) => Some(val.downcast_ref().cloned().unwrap()),
        None => None,
    }
}

// TODO: pass number of devices so the server waits for all of them ro register_device. Right now it works but idk why?
/// Starts the server used to sync the gradients of parameters sharded across multiple devices.
pub fn start_gradient_sync_server<B: Backend>() {
    log::info!("Starting gradient sync server.");
    match get_gradient_sync_client::<B>() {
        Some(_) => log::warn!("Client was already started"),
        None => {
            let typeid = TypeId::of::<B>();
            let mut state_map = get_backend_client_map();
            let client = GradientSyncClient::<B>::new();
            state_map.insert(typeid, Box::new(client.clone()));
        }
    }
}
