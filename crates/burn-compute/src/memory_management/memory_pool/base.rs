use super::index::SearchIndex;
use super::{
    ChunkHandle, ChunkId, MemoryChunk, MemoryPoolBinding, MemoryPoolHandle, MemorySlice,
    RingBuffer, SliceHandle, SliceId,
};
use crate::storage::{ComputeStorage, StorageHandle, StorageUtilization};
use hashbrown::{HashMap, HashSet};

pub struct MemoryPool {
    chunks: HashMap<ChunkId, Chunk>,
    slices: HashMap<SliceId, Slice>,
    #[allow(unused)] // will be used when we rewrite memory extension
    memory_extension_strategy: MemoryExtensionStrategy,
    rounding: RoundingStrategy,
    chunk_index: SearchIndex<ChunkId>,
    #[allow(unused)] // will be used when we rewrite memory extension
    max_chunk_size: usize,
    ring: RingBuffer<Chunk, Slice>,
}

#[derive(new, Debug)]
pub struct Chunk {
    pub storage: StorageHandle,
    pub handle: ChunkHandle,
    pub slices: MemoryPage,
}

// TODO: consider using generic trait and decouple from Slice
#[derive(new, Debug)]
pub struct MemoryPage {
    pub slices: HashMap<usize, SliceId>,
}

impl MemoryPage {
    /// merge slice at first_slice_address with the next slice (if there is one and if it's free)
    /// return a boolean representing if a merge happened
    fn merge_with_next_slice(
        &mut self,
        first_slice_address: usize,
        slices: &mut HashMap<SliceId, Slice>,
    ) -> bool {
        let first_slice_id = self.find_slice(first_slice_address).expect(
            "merge_with_next_slice shouldn't be called with a nonexistent first_slice address",
        );

        let next_slice_address =
            first_slice_address + slices.get(&first_slice_id).unwrap().effective_size();

        if let Some(next_slice_id) = self.find_slice(next_slice_address) {
            let (next_slice_eff_size, next_slice_is_free) = {
                let next_slice = slices.get(&next_slice_id).unwrap();
                (next_slice.effective_size(), next_slice.is_free())
            };
            if next_slice_is_free {
                let first_slice = slices.get_mut(&first_slice_id).unwrap();
                let first_slice_eff_size = first_slice.effective_size();
                let first_slice_offset = first_slice.storage.offset();

                let merged_size = first_slice_eff_size + next_slice_eff_size;
                first_slice.storage.utilization = StorageUtilization::Slice {
                    size: merged_size,
                    offset: first_slice_offset,
                };
                first_slice.padding = 0;

                // Cleanup of the extra slice
                self.slices.remove(&next_slice_address);
                slices.remove(&next_slice_id);
                return true;
            }
            return false;
        }
        false
    }

    fn find_slice(&self, address: usize) -> Option<SliceId> {
        let slice_id = self.slices.get(&address);
        slice_id.copied()
    }

    fn insert_slice(&mut self, address: usize, slice: &Slice) {
        self.slices.insert(address, slice.id());
    }
}

#[derive(new, Debug)]
pub struct Slice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub chunk: ChunkHandle,
    pub padding: usize,
}

impl Slice {
    pub fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }
}

const MIN_SIZE_NEEDED_TO_OFFSET: usize = 16;
const BUFFER_ALIGNMENT: usize = 32;
const MB: usize = 1024 * 1024;

pub enum RoundingStrategy {
    RoundUp,
    None,
}

impl RoundingStrategy {
    fn alloc_size(&self, size: usize) -> usize {
        match self {
            RoundingStrategy::RoundUp => {
                if size < BUFFER_ALIGNMENT {
                    return BUFFER_ALIGNMENT;
                }
                if size < MB {
                    2 * MB
                } else if size < 10 * MB {
                    return 20 * MB;
                } else {
                    let factor = (size + (2 * MB - 1)) / (2 * MB);
                    factor * 2 * MB
                }
            }
            RoundingStrategy::None => size,
        }
    }
}

/// The strategy defines the frequency at which merging of free slices (defragmentation) occurs
#[derive(Debug)]
pub enum MemoryExtensionStrategy {
    /// Once every n calls to reserve.
    PeriodTick {
        /// Number of calls to be executed before triggering the defragmentation.
        period: usize,
        /// Current state. Should start at zero.
        state: usize,
    },
    /// Never defragment.
    Never,
}

impl MemoryExtensionStrategy {
    /// Create a new strategy with the given period.
    pub fn new_period_tick(period: usize) -> Self {
        MemoryExtensionStrategy::PeriodTick { period, state: 0 }
    }

    #[allow(unused)]
    fn should_extend_max_memory(&mut self) -> bool {
        match self {
            MemoryExtensionStrategy::PeriodTick { period, state } => {
                *state = (*state + 1) % *period;
                *state == 0
            }
            MemoryExtensionStrategy::Never => false,
        }
    }
}

impl MemoryPool {
    pub fn new(
        merging_strategy: MemoryExtensionStrategy,
        alloc_strategy: RoundingStrategy,
        max_chunk_size: usize,
    ) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            memory_extension_strategy: merging_strategy,
            rounding: alloc_strategy,
            max_chunk_size,
            chunk_index: SearchIndex::new(),
            ring: RingBuffer::new(),
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
        if let Some(handle) = self.get_free_slice(size) {
            return MemoryPoolHandle { slice: handle };
        }

        let alloc_size = self.rounding.alloc_size(size);
        self.alloc_slice(storage, alloc_size, size)
    }

    fn alloc_slice<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_size: usize,
        slice_size: usize,
    ) -> MemoryPoolHandle {
        let handle_chunk = self.create_chunk(storage, alloc_size);
        let chunk_id = *handle_chunk.id();
        let (slice, extra_slice) =
            self.allocate_slices(handle_chunk.clone(), alloc_size, slice_size);

        let handle_slice = slice.handle.clone();
        self.update_chunk_metadata(chunk_id, slice, extra_slice);

        MemoryPoolHandle {
            slice: handle_slice,
        }
    }

    fn allocate_slices(
        &self,
        handle_chunk: ChunkHandle,
        alloc_size: usize,
        slice_size: usize,
    ) -> (Slice, Option<Slice>) {
        let slice = self.create_slice(0, slice_size, handle_chunk.clone());

        let effective_size = slice.effective_size();

        let extra_slice = if effective_size < alloc_size {
            Some(self.create_slice(effective_size, alloc_size - effective_size, handle_chunk))
        } else {
            None
        };

        (slice, extra_slice)
    }

    fn update_chunk_metadata(
        &mut self,
        chunk_id: ChunkId,
        slice: Slice,
        extra_slice: Option<Slice>,
    ) {
        let slice_id = *slice.handle.id();
        let slice_offset = slice.storage.offset();

        self.slices.insert(slice_id, slice);
        self.chunks
            .get_mut(&chunk_id)
            .unwrap()
            .slices
            .slices
            .insert(slice_offset, slice_id);

        if let Some(extra_slice) = extra_slice {
            let extra_slice_id = *extra_slice.handle.id();
            let extra_slice_offset = extra_slice.storage.offset();
            self.slices.insert(extra_slice_id, extra_slice);
            self.chunks
                .get_mut(&chunk_id)
                .unwrap()
                .slices
                .slices
                .insert(extra_slice_offset, extra_slice_id);
        }
    }

    #[allow(unused)]
    fn display_memory_usage(&self) {
        let total_memory_usage: f64 = self
            .chunks
            .values()
            .map(|chunk| chunk.storage.size() as f64)
            .sum();
        let effective_memory_usage: f64 = self
            .slices
            .values()
            .filter(|slice| slice.handle.is_free())
            .map(|slice| slice.storage.size() as f64)
            .sum();
        let ratio = 100.0 * effective_memory_usage / total_memory_usage;
        log::info!("the memory usage is {ratio}");
    }

    /// Finds a free slice that can contain the given size
    /// Returns the chunk's id and size.
    fn get_free_slice(&mut self, size: usize) -> Option<SliceHandle> {
        if size < MIN_SIZE_NEEDED_TO_OFFSET {
            return None;
        }

        let padding = calculate_padding(size);
        let effective_size = size + padding;

        let slice_id =
            self.ring
                .find_free_slice(effective_size, &mut self.chunks, &mut self.slices)?;

        let slice = self.slices.get_mut(&slice_id).unwrap();
        let old_slice_size = slice.effective_size();

        let offset = match slice.storage.utilization {
            StorageUtilization::Full(_) => 0,
            StorageUtilization::Slice { offset, size: _ } => offset,
        };
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
    fn create_slice(&self, offset: usize, size: usize, handle_chunk: ChunkHandle) -> Slice {
        assert_eq!(
            offset % BUFFER_ALIGNMENT,
            0,
            "slice with offset {offset} needs to be a multiple of {BUFFER_ALIGNMENT}"
        );
        if offset > 0 && size < MIN_SIZE_NEEDED_TO_OFFSET {
            panic!("tried to create slice of size {size} with an offset while the size needs to atleast be of size {MIN_SIZE_NEEDED_TO_OFFSET} for offset support");
        }
        let chunk = self.chunks.get(handle_chunk.id()).unwrap();
        let handle = SliceHandle::new();

        let storage = StorageHandle {
            id: chunk.storage.id.clone(),
            utilization: StorageUtilization::Slice { offset, size },
        };

        let padding = calculate_padding(size);

        Slice::new(storage, handle, chunk.handle.clone(), padding)
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

        self.ring.push_chunk(id);

        self.chunks.insert(
            id,
            Chunk::new(storage, handle.clone(), MemoryPage::new(HashMap::new())),
        );
        self.chunk_index.insert(id, size);

        handle
    }
    #[allow(unused)]
    fn deallocate<Storage: ComputeStorage>(
        &mut self,
        _storage: &mut Storage,
        _deallocations: &mut HashSet<ChunkId>,
    ) {
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

impl MemorySlice for Slice {
    fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    fn size(&self) -> usize {
        self.effective_size()
    }

    fn split(&mut self, offset_slice: usize) -> Option<Self> {
        let size_new = self.effective_size() - offset_slice;
        let offset_new = self.storage.offset() + offset_slice;
        let old_size = self.effective_size();

        let storage_new = StorageHandle {
            id: self.storage.id.clone(),
            utilization: StorageUtilization::Slice {
                offset: offset_new,
                size: size_new,
            },
        };

        self.storage.utilization = StorageUtilization::Slice {
            offset: self.storage.offset(),
            size: offset_slice,
        };

        if offset_new > 0 && size_new < MIN_SIZE_NEEDED_TO_OFFSET {
            panic!("tried to create slice of size {size_new} with an offset while the size needs to atleast be of size {MIN_SIZE_NEEDED_TO_OFFSET} for offset support");
        }
        if offset_new % BUFFER_ALIGNMENT != 0 {
            panic!("slice with offset {offset_new} needs to be a multiple of {BUFFER_ALIGNMENT}");
        }
        let handle = SliceHandle::new();
        if size_new < BUFFER_ALIGNMENT {
            self.padding = old_size - offset_slice;
            assert_eq!(self.effective_size(), old_size);
            return None;
        }

        assert!(
            size_new >= BUFFER_ALIGNMENT,
            "Size new > {BUFFER_ALIGNMENT}"
        );
        self.padding = 0;
        let padding = calculate_padding(size_new - BUFFER_ALIGNMENT);
        Some(Slice::new(storage_new, handle, self.chunk.clone(), padding))
    }

    fn id(&self) -> SliceId {
        *self.handle.id()
    }

    fn next_slice_position(&self) -> usize {
        self.storage.offset() + self.effective_size()
    }
}

impl MemoryChunk<Slice> for Chunk {
    fn merge_next_slice(
        &mut self,
        from_slice_index: usize,
        slices: &mut HashMap<SliceId, Slice>,
    ) -> bool {
        self.slices.merge_with_next_slice(from_slice_index, slices)
    }

    fn slice(&self, index: usize) -> Option<SliceId> {
        self.slices.find_slice(index)
    }

    fn insert_slice(
        &mut self,
        position: usize,
        slice: Slice,
        slices: &mut HashMap<SliceId, Slice>,
    ) {
        self.slices.insert_slice(position, &slice);
        slices.insert(slice.id(), slice);
    }
}
