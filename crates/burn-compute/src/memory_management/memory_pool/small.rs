use super::{ChunkHandle, ChunkId, MemoryPoolBinding, MemoryPoolHandle, SliceHandle, SliceId};
use crate::storage::{ComputeStorage, StorageHandle, StorageUtilization};
use hashbrown::HashMap;
use std::vec::Vec;

pub struct SmallMemoryPool {
    chunks: HashMap<ChunkId, SmallChunk>,
    slices: HashMap<SliceId, SmallSlice>,
    ring_buffer: Vec<ChunkId>,
    index: usize,
}

#[derive(new, Debug)]
pub struct SmallChunk {
    pub storage: StorageHandle,
    pub handle: ChunkHandle,
    pub slice: Option<SliceId>,
}

#[derive(new, Debug)]
pub struct SmallSlice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub chunk: ChunkHandle,
    pub padding: usize,
}

impl SmallSlice {
    pub fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }
}

const BUFFER_ALIGNMENT: usize = 32;

impl SmallMemoryPool {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            ring_buffer: Vec::new(),
            index: 0,
        }
    }

    /// Returns the resource from the storage, for the specified handle.
    #[allow(dead_code)]
    pub fn get<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        binding: &MemoryPoolBinding,
    ) -> Option<Storage::Resource> {
        self.slices
            .get(binding.slice.id())
            .map(|s| &s.storage)
            .map(|h| storage.get(h))
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    #[allow(dead_code)]
    pub fn reserve<Storage: ComputeStorage, Sync: FnOnce()>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        sync: Sync,
    ) -> MemoryPoolHandle {
        assert!(size <= BUFFER_ALIGNMENT);
        let slice = self.get_free_slice(size);

        match slice {
            Some(slice) => MemoryPoolHandle {
                slice: slice.clone(),
            },
            None => self.alloc(storage, size, sync),
        }
    }

    #[allow(dead_code)]
    pub fn alloc<Storage: ComputeStorage, Sync: FnOnce()>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        _sync: Sync,
    ) -> MemoryPoolHandle {
        assert!(size <= BUFFER_ALIGNMENT);
        if let Some(handle) = self.get_free_slice(size) {
            return MemoryPoolHandle { slice: handle };
        }

        self.alloc_slice(storage, size)
    }

    fn alloc_slice<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        slice_size: usize,
    ) -> MemoryPoolHandle {
        let handle_chunk = self.create_chunk(storage, BUFFER_ALIGNMENT);
        let chunk_id = *handle_chunk.id();
        let slice = self.allocate_slice(handle_chunk.clone(), slice_size);

        let handle_slice = slice.handle.clone();
        self.update_chunk_metadata(chunk_id, slice);

        MemoryPoolHandle {
            slice: handle_slice,
        }
    }

    fn allocate_slice(&self, handle_chunk: ChunkHandle, slice_size: usize) -> SmallSlice {
        let slice = self.create_slice(0, slice_size, handle_chunk.clone());

        let effective_size = slice.effective_size();
        assert_eq!(effective_size, BUFFER_ALIGNMENT);

        slice
    }

    fn update_chunk_metadata(&mut self, chunk_id: ChunkId, slice: SmallSlice) {
        let slice_id = *slice.handle.id();

        self.slices.insert(slice_id, slice);
        self.chunks.get_mut(&chunk_id).unwrap().slice = Some(slice_id);
    }

    fn find_free_slice(&mut self) -> Option<SliceId> {
        if self.ring_buffer.len() <= 0 {
            return None;
        }
        for _ in 0..self.ring_buffer.len() {
            let chunk_id = self.ring_buffer.get(self.index).unwrap();
            let chunk = self.chunks.get(chunk_id).unwrap();
            let slice = self.slices.get(&chunk.slice.unwrap()).unwrap();
            self.index = (self.index + 1) % self.ring_buffer.len();
            if slice.handle.is_free() {
                return Some(*slice.handle.id());
            }
        }
        None
    }

    /// Finds a free slice that can contain the given size
    /// Returns the chunk's id and size.
    fn get_free_slice(&mut self, size: usize) -> Option<SliceHandle> {
        let slice_id = self.find_free_slice();

        let slice_id = match slice_id {
            Some(val) => val,
            None => return None,
        };

        let slice = self.slices.get_mut(&slice_id).unwrap();
        let old_slice_size = slice.effective_size();

        let offset = match slice.storage.utilization {
            StorageUtilization::Full(_) => 0,
            StorageUtilization::Slice { offset, size: _ } => offset,
        };
        assert_eq!(offset, 0);
        slice.storage.utilization = StorageUtilization::Slice { offset, size };
        let new_padding = old_slice_size - size;
        slice.padding = new_padding;
        assert_eq!(
            slice.effective_size(),
            old_slice_size,
            "new and old slice should have the same size"
        );

        Some(slice.handle.clone())
    }

    /// Creates a slice of size `size` upon the given chunk with the given offset.
    fn create_slice(&self, offset: usize, size: usize, handle_chunk: ChunkHandle) -> SmallSlice {
        assert_eq!(offset, 0);
        let chunk = self.chunks.get(handle_chunk.id()).unwrap();
        let handle = SliceHandle::new();

        let storage = StorageHandle {
            id: chunk.storage.id.clone(),
            utilization: StorageUtilization::Slice { offset, size },
        };

        let padding = calculate_padding(size);

        SmallSlice::new(storage, handle, chunk.handle.clone(), padding)
    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
    ) -> ChunkHandle {
        let padding = calculate_padding(size);
        let effective_size = size + padding;

        let storage = storage.alloc(effective_size);
        let handle = ChunkHandle::new();
        let id = *handle.id();

        self.ring_buffer.push(id);

        self.chunks
            .insert(id, SmallChunk::new(storage, handle.clone(), None));

        handle
    }

    #[allow(unused)]
    fn deallocate<Storage: ComputeStorage>(&mut self, _storage: &mut Storage) {
        todo!()
    }
}

fn calculate_padding(size: usize) -> usize {
    let remainder = size % BUFFER_ALIGNMENT;
    if remainder != 0 {
        BUFFER_ALIGNMENT - remainder
    } else {
        0
    }
}
