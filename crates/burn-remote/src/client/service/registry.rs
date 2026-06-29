//! Process-global registry connecting Burn's compact `DeviceId` to rich remote endpoints.

use burn_ir::TensorId;
use burn_std::DeviceSettings;
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use super::conn::{EndpointKey, RemoteEndpoint};

static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn new_tensor_id() -> TensorId {
    TensorId::new(TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

struct EndpointRegistry {
    next_index: u32,
    by_endpoint: HashMap<(EndpointKey, u32), u32>,
    by_index: HashMap<u32, EndpointEntry>,
}

#[derive(Clone)]
struct EndpointEntry {
    endpoint: RemoteEndpoint,
    device_index: u32,
    settings: Arc<OnceLock<DeviceSettings>>,
    device_count: Arc<OnceLock<u32>>,
}

static REGISTRY: OnceLock<Mutex<EndpointRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<EndpointRegistry> {
    REGISTRY.get_or_init(|| {
        Mutex::new(EndpointRegistry {
            next_index: 0,
            by_endpoint: HashMap::new(),
            by_index: HashMap::new(),
        })
    })
}

pub(crate) fn register_endpoint(endpoint: RemoteEndpoint, device_index: u32) -> u32 {
    let key = (endpoint.key(), device_index);
    let mut registry = registry().lock().unwrap();
    if let Some(id) = registry.by_endpoint.get(&key).copied() {
        // Refresh mutable dialing hints while preserving the stable device id and settings cells.
        registry.by_index.get_mut(&id).unwrap().endpoint = endpoint;
        return id;
    }

    let id = registry.next_index;
    registry.next_index += 1;
    registry.by_endpoint.insert(key, id);
    registry.by_index.insert(
        id,
        EndpointEntry {
            endpoint,
            device_index,
            settings: Arc::new(OnceLock::new()),
            device_count: Arc::new(OnceLock::new()),
        },
    );
    id
}

pub(crate) fn endpoint_for(id: u32) -> Option<(RemoteEndpoint, u32)> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .map(|entry| (entry.endpoint.clone(), entry.device_index))
}

pub(crate) fn settings_for(id: u32) -> DeviceSettings {
    *settings_cell(id)
        .get()
        .expect("Remote service has not connected to this device yet")
}

pub(crate) fn has_settings(id: u32) -> bool {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .is_some_and(|entry| entry.settings.get().is_some())
}

pub(crate) fn settings_cell(id: u32) -> Arc<OnceLock<DeviceSettings>> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .expect("Device id not registered")
        .settings
        .clone()
}

pub(crate) fn device_count_cell(id: u32) -> Arc<OnceLock<u32>> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .expect("Device id not registered")
        .device_count
        .clone()
}

pub(crate) fn device_count_for(id: u32) -> Option<u32> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .and_then(|entry| entry.device_count.get().copied())
}
