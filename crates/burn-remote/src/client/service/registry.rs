//! Process-global registries shared across all remote clients.
//!
//! Two pieces of state outlive any single [`RemoteService`](super::RemoteService) and are
//! reachable without a handle to one:
//!
//! - a monotonic [`TensorId`] counter, so ids can be allocated on the calling thread
//!   without round-tripping to the device-runner thread (mirrors [`burn_fusion`]);
//! - an endpoint registry mapping an `(address, device index)` pair to a stable `u32` (the
//!   `index_id` carried by `RemoteDevice` → `DeviceId`) and to the cell holding the device's
//!   [`DeviceSettings`]. The cell is shared between `RemoteDevice::defaults` and
//!   `RemoteService::init` so the device can surface settings without holding the service.
//!   Two devices on the same host (same address, different device index) get distinct ids,
//!   so they land on distinct device-runner threads and connections.

use burn_ir::TensorId;
use burn_std::DeviceSettings;
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

/// Global monotonic tensor-id counter, shared across all remote clients.
static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Allocate a fresh, process-globally unique [`TensorId`].
pub(crate) fn new_tensor_id() -> TensorId {
    TensorId::new(TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

struct EndpointRegistry {
    next_index: u32,
    by_endpoint: HashMap<(String, u32), u32>,
    by_index: HashMap<u32, EndpointEntry>,
}

#[derive(Clone)]
struct EndpointEntry {
    address: String,
    device_index: u32,
    settings: Arc<OnceLock<DeviceSettings>>,
    /// Total number of devices hosted by the server at `address`, learned from the init
    /// handshake. A per-server property, but stored per id (every endpoint sharing the address
    /// observes the same value once any of them has connected). See [`device_count_for`].
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

/// Map an `(address, device index)` endpoint to a stable `u32` id (creating one if it's the
/// first time we see it).
///
/// Globally stable over the lifetime of the process; calling with the same endpoint always
/// returns the same id. The `address` should already be canonicalized (see
/// [`Address`](burn_communication::Address)) so equivalent spellings map to one id.
pub fn endpoint_to_id<S: AsRef<str>>(address: S, device_index: u32) -> u32 {
    let address = address.as_ref();
    let key = (address.to_string(), device_index);
    let mut reg = registry().lock().unwrap();
    if let Some(&id) = reg.by_endpoint.get(&key) {
        return id;
    }
    let id = reg.next_index;
    reg.next_index += 1;
    reg.by_endpoint.insert(key, id);
    reg.by_index.insert(
        id,
        EndpointEntry {
            address: address.to_string(),
            device_index,
            settings: Arc::new(OnceLock::new()),
            device_count: Arc::new(OnceLock::new()),
        },
    );
    id
}

/// Look up the `(address, device index)` endpoint bound to `id` by [`endpoint_to_id`].
pub fn id_to_endpoint(id: u32) -> Option<(String, u32)> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .map(|e| (e.address.clone(), e.device_index))
}

/// Returns the device settings registered for `id`.
///
/// Panics if no [`RemoteService`](super::RemoteService) has populated them yet (i.e., the
/// client has not been initialized for this device).
pub(crate) fn settings_for(id: u32) -> DeviceSettings {
    *settings_cell(id)
        .get()
        .expect("Remote service has not been initialized for this device yet")
}

/// Returns whether the device settings cell for `id` has been populated.
///
/// Used by `RemoteDevice::defaults` to decide whether a lazy-connect is needed before
/// reading.
pub(crate) fn has_settings(id: u32) -> bool {
    let reg = registry().lock().unwrap();
    reg.by_index
        .get(&id)
        .map(|e| e.settings.get().is_some())
        .unwrap_or(false)
}

/// The shared settings cell for `id`, populated once by `RemoteService::init` and read by
/// `RemoteDevice::defaults`.
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

/// The shared device-count cell for `id`, populated once by `RemoteService::init` from the
/// server's init handshake and read by [`device_count_for`].
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

/// Returns the number of devices the server hosts for `id`, or `None` if no
/// [`RemoteService`](super::RemoteService) has connected for this device yet.
pub(crate) fn device_count_for(id: u32) -> Option<u32> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .and_then(|e| e.device_count.get().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_to_id_is_stable_and_distinct() {
        let address1 = "ws://127.0.0.1:3000";
        let address2 = "ws://127.0.0.1:3001";

        let id1 = endpoint_to_id(address1, 0);
        let id2 = endpoint_to_id(address2, 0);

        assert_ne!(id1, id2);

        // Same endpoint always resolves to the same id, round-trips back to the endpoint.
        assert_eq!(endpoint_to_id(address1, 0), id1);
        assert_eq!(id_to_endpoint(id1), Some((address1.to_string(), 0)));

        assert_eq!(endpoint_to_id(address2, 0), id2);
        assert_eq!(id_to_endpoint(id2), Some((address2.to_string(), 0)));
    }

    #[test]
    fn endpoint_distinguishes_device_index_on_same_address() {
        let address = "ws://127.0.0.1:4000";

        let id0 = endpoint_to_id(address, 0);
        let id1 = endpoint_to_id(address, 1);

        // Same host, different device index → distinct ids → distinct service threads.
        assert_ne!(id0, id1);
        assert_eq!(id_to_endpoint(id0), Some((address.to_string(), 0)));
        assert_eq!(id_to_endpoint(id1), Some((address.to_string(), 1)));
    }

    #[test]
    fn id_to_endpoint_unknown_is_none() {
        assert_eq!(id_to_endpoint(u32::MAX), None);
    }

    #[test]
    fn new_tensor_id_is_monotonic() {
        let a = new_tensor_id();
        let b = new_tensor_id();
        assert_ne!(a, b);
    }

    #[test]
    fn settings_cell_is_shared_and_initially_empty() {
        let id = endpoint_to_id("ws://127.0.0.1:65000", 0);
        // No service has populated settings for a fresh id.
        assert!(!has_settings(id));

        // Repeated lookups hand back the *same* cell, so `RemoteDevice::defaults` and the
        // service observe each other's writes (and it starts empty).
        let cell_a = settings_cell(id);
        let cell_b = settings_cell(id);
        assert!(Arc::ptr_eq(&cell_a, &cell_b));
        assert!(cell_a.get().is_none());
    }
}
