use std::sync::Arc;

use burn_common::{reader::Reader, sync_type::SyncType};
use burn_compute::{
    memory_management::{simple::SimpleMemoryManagement, MemoryHandle, MemoryManagement},
    server::{Binding, ComputeServer, Handle},
    storage::{BytesResource, BytesStorage},
};
use derive_new::new;

use super::DummyKernel;

/// The dummy server is used to test the burn-compute infrastructure.
/// It uses simple memory management with a bytes storage on CPU, without asynchronous tasks.
#[derive(new, Debug)]
pub struct DummyServer<MM = SimpleMemoryManagement<BytesStorage>> {
    memory_management: MM,
}

impl<MM> ComputeServer for DummyServer<MM>
where
    MM: MemoryManagement<BytesStorage>,
{
    type Kernel = Arc<dyn DummyKernel>;
    type Storage = BytesStorage;
    type MemoryManagement = MM;
    type AutotuneKey = String;

    fn read(&mut self, binding: Binding<Self>) -> Reader<Vec<u8>> {
        let bytes = self.memory_management.get(binding.memory);

        Reader::Concrete(bytes.read().to_vec())
    }

    fn get_resource(&mut self, binding: Binding<Self>) -> BytesResource {
        self.memory_management.get(binding.memory)
    }

    fn create(&mut self, data: &[u8]) -> Handle<Self> {
        let handle = self.memory_management.reserve(data.len());
        let resource = self.memory_management.get(handle.clone().binding());

        let bytes = resource.write();

        for (i, val) in data.iter().enumerate() {
            bytes[i] = *val;
        }

        Handle::new(handle)
    }

    fn empty(&mut self, size: usize) -> Handle<Self> {
        Handle::new(self.memory_management.reserve(size))
    }

    fn execute(&mut self, kernel: Self::Kernel, bindings: Vec<Binding<Self>>) {
        let mut resources = bindings
            .into_iter()
            .map(|binding| self.memory_management.get(binding.memory))
            .collect::<Vec<_>>();

        kernel.compute(&mut resources);
    }

    fn sync(&mut self, _: SyncType) {
        // Nothing to do with dummy backend.
    }
}
