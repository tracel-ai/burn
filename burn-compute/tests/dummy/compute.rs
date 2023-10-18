use super::DummyServer;
use burn_compute::channel::MutexComputeChannel;
use burn_compute::client::ComputeClient;
use burn_compute::memory_management::{DeallocStrategy, SimpleMemoryManagement, SliceStrategy};
use burn_compute::storage::BytesStorage;
use burn_compute::Compute;

/// The dummy device.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyDevice;

pub type DummyChannel = MutexComputeChannel<DummyServer>;
pub type DummyClient = ComputeClient<DummyServer, DummyChannel>;

static COMPUTE: Compute<DummyDevice, DummyServer, DummyChannel> = Compute::new();

pub fn client(device: &DummyDevice) -> DummyClient {
    COMPUTE.client(device, || {
        let storage = BytesStorage::default();
        let memory_management =
            SimpleMemoryManagement::new(storage, DeallocStrategy::Never, SliceStrategy::Never);
        let server = DummyServer::new(memory_management);
        let channel = MutexComputeChannel::new(server);

        ComputeClient::new(channel)
    })
}
