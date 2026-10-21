use std::{
    any::{Any, TypeId},
    sync::{Mutex, MutexGuard, OnceLock},
};

use std::collections::{HashMap, hash_map::Entry};

use crate::Backend;

use super::{DistributedConfig, client::DistributedSyncClient};

/// The type-erased box type for [`DistributedSyncClient`].
type ClientBox = Box<dyn Any + Send + Sync>;

/// Global state map from [Backend] to boxed [`DistributedSyncClient`].
static BACKEND_CLIENT_MAP: OnceLock<Mutex<HashMap<TypeId, ClientBox>>> = OnceLock::new();

// TODO: Replace TypeId with DeviceId, the index being i32::MAX, a.k.a. communication index.
/// Gets a locked mutable view of the `STATE_MAP`.
pub(crate) fn get_backend_client_map() -> MutexGuard<'static, HashMap<TypeId, ClientBox>> {
    BACKEND_CLIENT_MAP
        .get_or_init(Default::default)
        .lock()
        .unwrap()
}

/// Get the distributed sync client for the given [Backend].
pub fn get_distributed_sync_client<B: Backend>() -> Option<DistributedSyncClient<B>> {
    let typeid = TypeId::of::<B>();
    let state_map = get_backend_client_map();
    state_map
        .get(&typeid)
        .map(|val| val.downcast_ref().cloned().unwrap())
}

/// Remove the client form the map for the given [Backend].
pub(crate) fn remove_distributed_sync_client<B: Backend>() {
    let typeid = TypeId::of::<B>();
    let mut state_map = get_backend_client_map();
    state_map.remove(&typeid);
}

/// Starts the server used to sync the gradients of parameters sharded across multiple devices.
///
/// # Panics
///
/// When one already runs for `B`: there is one per backend, and the devices of a second would sync
/// against the first one's.
pub fn start_distributed_sync_server<B: Backend>(devices: &[B::Device], config: DistributedConfig) {
    let started = match get_backend_client_map().entry(TypeId::of::<B>()) {
        Entry::Occupied(_) => false,
        Entry::Vacant(entry) => {
            entry.insert(Box::new(DistributedSyncClient::<B>::new(devices, config)));
            true
        }
    };
    // Outside the lock, which a panic would poison for every later server.
    assert!(
        started,
        "A distributed context already runs for this backend: drop it before starting another"
    );
}

/// Close the gradient syncing server.
pub fn close_distributed_sync_server<B: Backend>() {
    if let Some(client) = get_distributed_sync_client::<B>() {
        remove_distributed_sync_client::<B>();
        client.close();
    }
}
