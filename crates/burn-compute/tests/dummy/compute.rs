use std::sync::Arc;

use super::DummyServer;
use burn_common::stub::RwLock;
use burn_compute::channel::MutexComputeChannel;
use burn_compute::client::ComputeClient;
use burn_compute::memory_management::simple::{
    DeallocStrategy, SimpleMemoryManagement, SliceStrategy,
};
use burn_compute::storage::BytesStorage;
use burn_compute::tune::Tuner;
use burn_compute::ComputeRuntime;

/// The dummy device.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyDevice;

pub type DummyChannel = MutexComputeChannel<DummyServer>;
pub type DummyClient = ComputeClient<DummyServer, DummyChannel>;

static RUNTIME: ComputeRuntime<DummyDevice, DummyServer, DummyChannel> = ComputeRuntime::new();
pub static TUNER_DEVICE_ID: &str = "tests/dummy-device";
pub static TUNER_PREFIX: &str = "dummy-tests/dummy-device";

pub fn init_client() -> ComputeClient<DummyServer, MutexComputeChannel<DummyServer>> {
    let storage = BytesStorage::default();
    let memory_management =
        SimpleMemoryManagement::new(storage, DeallocStrategy::Never, SliceStrategy::Never);
    let server = DummyServer::new(memory_management);
    let channel = MutexComputeChannel::new(server);
    let tuner = Arc::new(RwLock::new(Tuner::new("dummy", TUNER_DEVICE_ID)));
    ComputeClient::new(channel, tuner)
}

pub fn client(device: &DummyDevice) -> DummyClient {
    RUNTIME.client(device, init_client)
}
