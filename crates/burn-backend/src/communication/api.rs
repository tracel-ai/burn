use std::{
    any::{Any, TypeId},
    sync::{Mutex, MutexGuard, OnceLock},
};

use hashbrown::HashMap;

use crate::{Backend, client::GradientSyncClient};

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

// Device service/id type u16 max ____ index max u32.

/// Gets a locked mutable view of the `STATE_MAP`.
pub(crate) fn get_backend_client_map() -> MutexGuard<'static, HashMap<TypeId, ClientBox>> {
    BACKEND_CLIENT_MAP
        .get_or_init(Default::default)
        .lock()
        .unwrap()
}

/// Get a [`GradientSyncClient`] for the given [`Backend`].
pub fn get_gradient_sync_client<B: Backend>(device: &B::Device) -> Option<GradientSyncClient<B>> {
    let typeid = TypeId::of::<B>();
    // let typeid = B::type_id(device);
    println!("type_id get: {:?}", typeid);
    let state_map = get_backend_client_map();
    println!("map : {:?}", state_map);
    match state_map.get(&typeid) {
        Some(val) => Some(val.downcast_ref().cloned().unwrap()),
        None => None,
    }
}

/// Remove the client form the map for the given [`Backend`].
pub(crate) fn remove_gradient_sync_client<B: Backend>(device: &B::Device) {
    // let typeid = TypeId::of::<B>();
    let typeid = B::type_id(device);
    let mut state_map = get_backend_client_map();
    state_map.remove(&typeid);
}

/// Starts the server used to sync the gradients of parameters sharded across multiple devices.
pub fn start_gradient_sync_server<B: Backend>(devices: Vec<B::Device>) {
    if get_gradient_sync_client::<B>(&devices[0]).is_none() {
        let typeid = TypeId::of::<B>();
        // let typeid = B::type_id(&devices[0]);
        println!("type_id start: {:?}", typeid);
        let mut state_map = get_backend_client_map();
        // let client = match typeid {
        //     TypeId::of::<Cuda>() => GradientSyncClient::<Cuda>::new(devices),
        // };
        let client = GradientSyncClient::<B>::new(devices);
        state_map.insert(typeid, Box::new(client.clone()));
    }
}

/// Close the gradient syncing server.
pub fn close_gradient_sync_server<B: Backend>(device: &B::Device) {
    if let Some(client) = get_gradient_sync_client::<B>(device) {
        client.close();
        remove_gradient_sync_client::<B>(device);
    }
}
