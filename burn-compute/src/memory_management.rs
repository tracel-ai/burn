use alloc::vec::Vec;
use derive_new::new;

pub enum Memory<'a> {
    MemoryChunk(MemoryChunk),
    MemorySlice(MemorySlice<'a>),
}

impl Memory<'_> {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Memory::MemoryChunk(chunk) => chunk.chunk.clone(),
            Memory::MemorySlice(slice) => {
                slice.chunk[slice.location..slice.location + slice.size].into()
            }
        }
    }

    pub fn write(&mut self, bytes: Vec<u8>) {
        match self {
            Memory::MemoryChunk(chunk) => chunk.chunk = bytes,
            Memory::MemorySlice(slice) => (), // ??
        }
    }
}

#[derive(new)]
pub struct MemorySlice<'a> {
    chunk: &'a Vec<u8>,
    location: usize,
    size: usize,
}

#[derive(new)]
pub struct MemoryChunk {
    chunk: Vec<u8>,
}

pub struct MemoryDescription {
    pub memory_id: usize,
}

pub trait MemoryManagement {
    type Allocator; // In WGPU, this would be Device

    fn get(&self, description: MemoryDescription) -> Memory;
    fn init(&mut self, resource: Vec<u8>) -> MemoryDescription;
}
