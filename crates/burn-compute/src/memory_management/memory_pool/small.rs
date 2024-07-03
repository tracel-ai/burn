use super::{ChunkHandle, ChunkId, MemoryPoolBinding, MemoryPoolHandle, SliceHandle, SliceId};
use crate::storage::{ComputeStorage, StorageHandle, StorageUtilization};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// A memory pool that allocates fixed-size chunks (32 bytes each) and reuses them to minimize allocations.
///
/// - Only one slice is supported per chunk due to the limitations in WGPU where small allocations cannot be offset.
/// - The pool uses a ring buffer to efficiently manage and reuse chunks.
///
/// Fields:
/// - `chunks`: A hashmap storing the allocated chunks by their IDs.
/// - `slices`: A hashmap storing the slices by their IDs.
/// - `ring_buffer`: A vector used as a ring buffer to manage chunk reuse.
/// - `index`: The current position in the ring buffer.
pub struct SmallMemoryPool {
    chunks: HashMap<ChunkId, SmallChunk>,
    slices: HashMap<SliceId, SmallSlice>,
    ring_buffer: Vec<ChunkId>,
    index: usize,
    buffer_storage_alignment_offset: usize,
}

#[derive(new, Debug)]
pub struct SmallChunk {
    pub storage: StorageHandle,
    #[allow(dead_code)]
    pub handle: ChunkHandle,
    pub slice: Option<SliceId>,
}

#[derive(new, Debug)]
pub struct SmallSlice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    #[allow(dead_code)]
    pub chunk: ChunkHandle,
    pub padding: usize,
}

impl SmallSlice {
    pub fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }
}

impl SmallMemoryPool {
    pub fn new(buffer_storage_alignment_offset: usize) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            ring_buffer: Vec::new(),
            index: 0,
            buffer_storage_alignment_offset,
        }
    }

    /// Returns the resource from the storage, for the specified handle.
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
    pub fn reserve<Storage: ComputeStorage, Sync: FnOnce()>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        sync: Sync,
    ) -> MemoryPoolHandle {
        assert!(size <= self.buffer_storage_alignment_offset);
        let slice = self.get_free_slice(size);

        match slice {
            Some(slice) => MemoryPoolHandle {
                slice: slice.clone(),
            },
            None => self.alloc(storage, size, sync),
        }
    }

    pub fn alloc<Storage: ComputeStorage, Sync: FnOnce()>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        _sync: Sync,
    ) -> MemoryPoolHandle {
        assert!(size <= self.buffer_storage_alignment_offset);

        self.alloc_slice(storage, size)
    }

    fn alloc_slice<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        slice_size: usize,
    ) -> MemoryPoolHandle {
        let handle_chunk = self.create_chunk(storage, self.buffer_storage_alignment_offset);
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
        assert_eq!(effective_size, self.buffer_storage_alignment_offset);

        slice
    }

    fn update_chunk_metadata(&mut self, chunk_id: ChunkId, slice: SmallSlice) {
        let slice_id = *slice.handle.id();

        self.slices.insert(slice_id, slice);
        self.chunks.get_mut(&chunk_id).unwrap().slice = Some(slice_id);
    }

    fn find_free_slice(&mut self) -> Option<SliceId> {
        if self.ring_buffer.is_empty() {
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
        let slice_id = self.find_free_slice()?;

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

        let padding = calculate_padding(size, self.buffer_storage_alignment_offset);

        SmallSlice::new(storage, handle, chunk.handle.clone(), padding)
    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
    ) -> ChunkHandle {
        let padding = calculate_padding(size, self.buffer_storage_alignment_offset);
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

fn calculate_padding(size: usize, buffer_storage_alignment_offset: usize) -> usize {
    let remainder = size % buffer_storage_alignment_offset;
    if remainder != 0 {
        buffer_storage_alignment_offset - remainder
    } else {
        0
    }
}
