use spin::Mutex;

use crate::channel::MutexComputeChannel;
use crate::client::ComputeClient;
use crate::compute::Compute;
use crate::tests::integration::dummy_server::DummyServer;
use crate::{BytesStorage, ComputeChannel, SimpleMemoryManagement};

/// The dummy device.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyDevice;

static COMPUTE: Mutex<Compute<DummyDevice, DummyServer>> = Mutex::new(Compute::new());

pub fn get(device: &DummyDevice) -> ComputeClient<DummyServer> {
    let mut compute = COMPUTE.lock();

    compute.get(device, || {
        let storage = BytesStorage::default();
        let memory_management = SimpleMemoryManagement::never_dealloc(storage);
        let server = DummyServer::new(memory_management);
        let channel = MutexComputeChannel::new(server);

        ComputeClient::new(channel)
    })
}
