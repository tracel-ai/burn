//! Process-global registries shared across all remote clients.
//!
//! Two pieces of state outlive any single [`RemoteService`](super::RemoteService) and are
//! reachable without a handle to one:
//!
//! - a monotonic [`TensorId`] counter, so ids can be allocated on the calling thread
//!   without round-tripping to the device-runner thread (mirrors [`burn_fusion`]);
//! - an address registry mapping a network address to a stable `u32` (the `index_id`
//!   carried by `RemoteDevice` → `DeviceId`) and to the cell holding the device's
//!   [`DeviceSettings`]. The cell is shared between `RemoteDevice::defaults` and
//!   `RemoteService::init` so the device can surface settings without holding the service.

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

struct AddressRegistry {
    next_index: u32,
    by_address: HashMap<String, u32>,
    by_index: HashMap<u32, AddressEntry>,
}

#[derive(Clone)]
struct AddressEntry {
    address: String,
    settings: Arc<OnceLock<DeviceSettings>>,
}

static REGISTRY: OnceLock<Mutex<AddressRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<AddressRegistry> {
    REGISTRY.get_or_init(|| {
        Mutex::new(AddressRegistry {
            next_index: 0,
            by_address: HashMap::new(),
            by_index: HashMap::new(),
        })
    })
}

/// Map a network address to a stable `u32` id (creating one if it's the first time we see it).
///
/// Globally stable over the lifetime of the process; calling with the same address always
/// returns the same id.
pub fn address_to_id<S: AsRef<str>>(address: S) -> u32 {
    let address = address.as_ref();
    let mut reg = registry().lock().unwrap();
    if let Some(&id) = reg.by_address.get(address) {
        return id;
    }
    let id = reg.next_index;
    reg.next_index += 1;
    reg.by_address.insert(address.to_string(), id);
    reg.by_index.insert(
        id,
        AddressEntry {
            address: address.to_string(),
            settings: Arc::new(OnceLock::new()),
        },
    );
    id
}

/// Look up the address bound to `id` by [`address_to_id`].
pub fn id_to_address(id: u32) -> Option<String> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .map(|e| e.address.clone())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_to_id_is_stable_and_distinct() {
        let address1 = "ws://127.0.0.1:3000";
        let address2 = "ws://127.0.0.1:3001";

        let id1 = address_to_id(address1);
        let id2 = address_to_id(address2);

        assert_ne!(id1, id2);

        // Same address always resolves to the same id, round-trips back to the address.
        assert_eq!(address_to_id(address1), id1);
        assert_eq!(id_to_address(id1), Some(address1.to_string()));

        assert_eq!(address_to_id(address2), id2);
        assert_eq!(id_to_address(id2), Some(address2.to_string()));
    }

    #[test]
    fn id_to_address_unknown_is_none() {
        assert_eq!(id_to_address(u32::MAX), None);
    }

    #[test]
    fn new_tensor_id_is_monotonic() {
        let a = new_tensor_id();
        let b = new_tensor_id();
        assert_ne!(a, b);
    }

    #[test]
    fn settings_cell_is_shared_and_initially_empty() {
        let id = address_to_id("ws://127.0.0.1:65000");
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
