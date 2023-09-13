// use crate::{Allocator, Memory, MemoryChunk, MemoryId, MemoryManagement};
// use alloc::sync::Arc;
// use alloc::vec::Vec;
// use core::cell::Cell;
// use core::sync::atomic::AtomicUsize;
//
// type DummyMemory = Cell<Vec<u8>>;
//
// #[derive(Default)]
// pub struct DummyMemoryManagement;
//
// pub struct MemorySliceMetadata {
//     start: AtomicUsize,
//     end: AtomicUsize,
// }
//
// impl MemoryManagement for DummyMemoryManagement {
//     type MemoryId = DummyMemoryId;
//     type Allocator = BytesAllocator;
//
//     fn get<'a>(&'a self, description: &Self::MemoryId) -> Memory<'a, MemoryChunk<Self::Allocator>> {
//         todo!();
//         //Memory::MemoryChunk(self.storage.get(*description).unwrap())
//     }
//
//     fn init(&mut self, resource: Vec<u8>) -> Self::MemoryId {
//         todo!();
//         //self.list_of_sizes.push(resource.len());
//         //self.storage.push(Cell::new(resource));
//         //let index = self.list_of_sizes.len() - 1;
//
//         //DummyMemoryId {
//         //    storage_location: index,
//         //}
//     }
//
//     fn empty(&mut self, size: usize) -> Self::MemoryId {
//         todo!();
//         // let bytes = self.allocator.alloc(size);
//         // self.list_of_sizes.push(size);
//
//         // self.storage.push(bytes);
//         // let index = self.list_of_sizes.len() - 1;
//
//         // DummyMemoryId {
//         //     storage_location: index,
//         // }
//     }
// }
