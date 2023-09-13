use alloc::vec::Vec;
use derive_new::new;

use super::DummyKernelDescription;
use crate::{BasicMemoryManagement, BytesStorage, ComputeServer, MemoryManagement, ServerResource};

#[derive(new)]
pub struct DummyServer<MM = BasicMemoryManagement<BytesStorage>> {
    memory_management: MM,
}

impl<MM> ComputeServer for DummyServer<MM>
where
    MM: MemoryManagement<BytesStorage>,
{
    type KernelDescription = DummyKernelDescription;
    type Storage = BytesStorage;
    type MemoryManagement = MM;

    fn read(&mut self, resource_description: &ServerResource<Self>) -> Vec<u8> {
        let bytes = self.memory_management.get(resource_description);

        bytes.read().to_vec()
    }

    fn create(&mut self, data: Vec<u8>) -> ServerResource<Self> {
        let resource = self.memory_management.reserve(data.len());
        let bytes = self.memory_management.get(&resource);

        let bytes = bytes.write();

        for (i, val) in data.into_iter().enumerate() {
            bytes[i] = val;
        }

        resource
    }

    fn empty(&mut self, size: usize) -> ServerResource<Self> {
        self.memory_management.reserve(size)
    }

    fn execute(
        &mut self,
        kernel_description: Self::KernelDescription,
        resource_descriptions: &[&ServerResource<Self>],
    ) {
        let mut resources = resource_descriptions
            .iter()
            .map(|r| self.memory_management.get(&r))
            .collect::<Vec<_>>();

        kernel_description.compute(&mut resources);
    }

    fn sync(&self) {
        // Nothing to do with dummy backend.
    }
}
