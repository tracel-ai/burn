use std::mem;

use alloc::{boxed::Box, vec, vec::Vec};
use derive_new::new;

use crate::{ComputeServer, Memory, MemoryManagement};

use super::{DummyAllocator, DummyKernelDescription};

#[derive(new)]
pub struct DummyServer {
    memory_management: Box<dyn MemoryManagement<Allocator = DummyAllocator>>,
}
#[derive(new)]
pub struct DummyResourceDescription {
    memory_id: usize,
}

impl ComputeServer for DummyServer {
    type KernelDescription = DummyKernelDescription;

    type ResourceDescription = DummyResourceDescription;

    fn read(&mut self, resource_description: &Self::ResourceDescription) -> Vec<u8> {
        self.get_resource(resource_description).to_bytes()
    }

    fn create(&mut self, resource: Vec<u8>) -> Self::ResourceDescription {
        let memory_description = self.memory_management.init(resource);
        DummyResourceDescription::new(memory_description.memory_id)
    }

    fn empty(&mut self, size: usize) -> Self::ResourceDescription {
        self.create(vec![0; size])
    }

    fn execute(
        &mut self,
        kernel_description: Self::KernelDescription,
        resource_descriptions: Vec<&Self::ResourceDescription>,
    ) {
        // todo: queue kernels

        let memories: Vec<Memory<'_>> = resource_descriptions
            .iter()
            .map(|rd| self.get_resource(rd))
            .collect();
        kernel_description.compute(memories)
    }

    fn sync(&self) {
        todo!()
    }
}

impl DummyServer {
    fn get_resource(&self, resource_description: &DummyResourceDescription) -> Memory {
        self.memory_management.get(crate::MemoryDescription {
            memory_id: resource_description.memory_id,
        })
    }
}
