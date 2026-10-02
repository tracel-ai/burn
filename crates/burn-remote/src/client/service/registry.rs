//! Process-global registry connecting Burn's compact `DeviceId` to rich remote endpoints.

use burn_ir::TensorId;
use burn_std::DeviceSettings;
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use super::conn::{EndpointKey, RemoteEndpoint};

static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn new_tensor_id() -> TensorId {
    TensorId::new(TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Whether a device's session has ended: its response stream closed, or its writer failed.
///
/// An ended session is never reopened in place. Its tensors, fused graphs and settings belong to
/// a server session that is gone, so the next connect registers a new device instead.
#[derive(Default)]
pub(crate) struct SessionState {
    ended: AtomicBool,
}

impl SessionState {
    pub(crate) fn end(&self) {
        self.ended.store(true, Ordering::Release);
    }

    pub(crate) fn has_ended(&self) -> bool {
        self.ended.load(Ordering::Acquire)
    }
}

struct EndpointRegistry {
    next_index: u32,
    /// The device id currently serving each endpoint and device index.
    current: HashMap<(EndpointKey, u32), u32>,
    by_index: HashMap<u32, EndpointEntry>,
}

struct EndpointEntry {
    endpoint: RemoteEndpoint,
    device_index: u32,
    settings: Arc<OnceLock<DeviceSettings>>,
    device_count: Arc<OnceLock<u32>>,
    session: Arc<SessionState>,
}

static REGISTRY: OnceLock<Mutex<EndpointRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<EndpointRegistry> {
    REGISTRY.get_or_init(|| {
        Mutex::new(EndpointRegistry {
            next_index: 0,
            current: HashMap::new(),
            by_index: HashMap::new(),
        })
    })
}

/// The device id for `endpoint` and `device_index`: the current one, with its dialing hints
/// refreshed, or a new one when there is none or its session has ended.
pub(crate) fn register_endpoint(endpoint: RemoteEndpoint, device_index: u32) -> u32 {
    let key = (endpoint.key(), device_index);
    let mut registry = registry().lock().unwrap();
    if let Some(id) = registry.current.get(&key).copied() {
        let entry = registry.by_index.get_mut(&id).unwrap();
        if !entry.session.has_ended() {
            entry.endpoint = endpoint;
            return id;
        }
    }

    let id = registry.next_index;
    // A `DeviceId` carries the registry id in 16 bits. The lock is released first: a panic with
    // it held would poison the registry for every device already connected.
    if id > u32::from(u16::MAX) {
        drop(registry);
        panic!(
            "Burn Remote has registered {id} remote devices in this process, more than a device id \
             can name"
        );
    }
    registry.next_index += 1;
    registry.current.insert(key, id);
    registry.by_index.insert(
        id,
        EndpointEntry {
            endpoint,
            device_index,
            settings: Arc::new(OnceLock::new()),
            device_count: Arc::new(OnceLock::new()),
            session: Arc::new(SessionState::default()),
        },
    );
    id
}

/// A registered device: where its server is, and which of the server's devices it is.
pub(crate) struct RegisteredDevice {
    pub(crate) endpoint: RemoteEndpoint,
    pub(crate) device_index: u32,
}

fn with_entry<T>(id: u32, read: impl FnOnce(&EndpointEntry) -> T) -> T {
    let registry = registry().lock().unwrap();
    read(
        registry
            .by_index
            .get(&id)
            .expect("Device id not registered"),
    )
}

pub(crate) fn registered_device(id: u32) -> Option<RegisteredDevice> {
    registry()
        .lock()
        .unwrap()
        .by_index
        .get(&id)
        .map(|entry| RegisteredDevice {
            endpoint: entry.endpoint.clone(),
            device_index: entry.device_index,
        })
}

pub(crate) fn settings_for(id: u32) -> DeviceSettings {
    *settings_cell(id)
        .get()
        .expect("Remote service has not connected to this device yet")
}

pub(crate) fn has_settings(id: u32) -> bool {
    with_entry(id, |entry| entry.settings.get().is_some())
}

pub(crate) fn settings_cell(id: u32) -> Arc<OnceLock<DeviceSettings>> {
    with_entry(id, |entry| entry.settings.clone())
}

pub(crate) fn device_count_cell(id: u32) -> Arc<OnceLock<u32>> {
    with_entry(id, |entry| entry.device_count.clone())
}

pub(crate) fn device_count_for(id: u32) -> Option<u32> {
    with_entry(id, |entry| entry.device_count.get().copied())
}

pub(crate) fn session_state(id: u32) -> Arc<SessionState> {
    with_entry(id, |entry| entry.session.clone())
}

pub(crate) fn session_ended(id: u32) -> bool {
    with_entry(id, |entry| entry.session.has_ended())
}
