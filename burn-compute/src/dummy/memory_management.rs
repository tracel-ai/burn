use alloc::vec::Vec;
use derive_new::new;

use crate::{Memory, MemoryChunk, MemoryDescription, MemoryManagement};

#[derive(new)]
pub struct DummyMemoryManagement {
    list_of_sizes: Vec<usize>,
    allocator: DummyAllocator,
}

pub struct DummyAllocator {
    storage: Vec<Vec<u8>>,
}

impl DummyAllocator {
    pub fn new() -> Self {
        DummyAllocator {
            storage: Vec::new(),
        }
    }

    pub fn allocate(&mut self, resource: Vec<u8>) {
        self.storage.push(resource);
    }
}

impl MemoryManagement for DummyMemoryManagement {
    type Allocator = DummyAllocator;

    fn get(&self, description: MemoryDescription) -> Memory {
        Memory::MemoryChunk(MemoryChunk::new(
            self.allocator
                .storage
                .get(description.memory_id)
                .unwrap()
                .clone(),
        ))
    }

    fn init(&mut self, resource: Vec<u8>) -> MemoryDescription {
        self.list_of_sizes.push(resource.len());
        self.allocator.allocate(resource);
        MemoryDescription {
            memory_id: self.list_of_sizes.len() - 1,
        }
    }
}
