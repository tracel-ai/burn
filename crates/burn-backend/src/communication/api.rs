use std::{
    any::{Any, TypeId},
    sync::{Mutex, MutexGuard, OnceLock},
};

use hashbrown::HashMap;

use crate::{Backend, DistributedConfig, client::DistributedSyncClient};

/// The type-erased box type for [`DistributedSyncClient`].
type ClientBox = Box<dyn Any + Send + Sync>;

/// Global state map from [`Backend`] to boxed [`DistributedSyncClient`].
static BACKEND_CLIENT_MAP: OnceLock<Mutex<HashMap<TypeId, ClientBox>>> = OnceLock::new();

// TODO: Replace TypeId with DeviceId, the index being i32::MAX, a.k.a. communication index.
/// Gets a locked mutable view of the `STATE_MAP`.
pub(crate) fn get_backend_client_map() -> MutexGuard<'static, HashMap<TypeId, ClientBox>> {
    BACKEND_CLIENT_MAP
        .get_or_init(Default::default)
        .lock()
        .unwrap()
}

/// Get a [`DistributedSyncClient`] for the given [`Backend`].
pub fn get_distributed_sync_client<B: Backend>() -> Option<DistributedSyncClient<B>> {
    let typeid = TypeId::of::<B>();
    let state_map = get_backend_client_map();
    match state_map.get(&typeid) {
        Some(val) => Some(val.downcast_ref().cloned().unwrap()),
        None => None,
    }
}

/// Remove the client form the map for the given [`Backend`].
pub(crate) fn remove_distributed_sync_client<B: Backend>() {
    let typeid = TypeId::of::<B>();
    let mut state_map = get_backend_client_map();
    state_map.remove(&typeid);
}

/// Starts the server used to sync the gradients of parameters sharded across multiple devices.
pub fn start_distributed_sync_server<B: Backend>(
    devices: Vec<B::Device>,
    config: DistributedConfig,
) {
    if get_distributed_sync_client::<B>().is_none() {
        let typeid = TypeId::of::<B>();
        let mut state_map = get_backend_client_map();
        let client = DistributedSyncClient::<B>::new(devices.len(), config);
        state_map.insert(typeid, Box::new(client.clone()));
    }
}

/// Close the gradient syncing server.
pub fn close_distributed_sync_server<B: Backend>() {
    if let Some(client) = get_distributed_sync_client::<B>() {
        client.close();
        remove_distributed_sync_client::<B>();
    }
}
