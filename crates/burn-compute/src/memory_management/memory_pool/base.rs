use super::index::SearchIndex;
use super::{
    ChunkHandle, ChunkId, MemoryChunk, MemoryPoolBinding, MemoryPoolHandle, MemorySlice,
    RingBuffer, SliceHandle, SliceId,
};
use crate::storage::{ComputeStorage, StorageHandle, StorageUtilization};
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

pub struct MemoryPool {
    chunks: HashMap<ChunkId, Chunk>,
    slices: HashMap<SliceId, Slice>,
    memory_extension_strategy: MemoryExtensionStrategy,
    rounding: RoundingStrategy,
    chunk_index: SearchIndex<ChunkId>,
    max_chunk_size: usize,
    ring: RingBuffer<Chunk, Slice>,
    debug: bool,
}

#[derive(new, Debug)]
pub struct Chunk {
    pub storage: StorageHandle,
    pub handle: ChunkHandle,
    pub slices: Vec<SliceId>,
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

struct SliceUpdate {
    slice_id: SliceId,
    size: usize,
}

impl MemoryPool {
    pub fn new(
        merging_strategy: MemoryExtensionStrategy,
        alloc_strategy: RoundingStrategy,
        max_chunk_size: usize,
        debug: bool,
    ) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            memory_extension_strategy: merging_strategy,
            rounding: alloc_strategy,
            max_chunk_size,
            chunk_index: SearchIndex::new(),
            ring: RingBuffer::new(),
            debug,
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
        // Looks for a large enough, existing but unused chunk of memory.
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
        sync: Sync,
    ) -> MemoryPoolHandle {
        if self.memory_extension_strategy.should_extend_max_memory() {
            sync();
            self.extend_max_memory(storage, size);

            if let Some(handle) = self.get_free_slice(size) {
                return MemoryPoolHandle { slice: handle };
            }
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

        self.slices.insert(slice_id, slice);
        self.chunks
            .get_mut(&chunk_id)
            .unwrap()
            .slices
            .push(slice_id);

        if let Some(extra_slice) = extra_slice {
            let extra_slice_id = *extra_slice.handle.id();
            self.slices.insert(extra_slice_id, extra_slice);
            self.chunks
                .get_mut(&chunk_id)
                .unwrap()
                .slices
                .push(extra_slice_id);
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
                .find_free_slice(effective_size, &mut self.chunks, &mut self.slices);

        let slice_id = match slice_id {
            Some(val) => val,
            None => return None,
        };

        let slice = self.slices.get_mut(&slice_id).unwrap();

        let offset = match slice.storage.utilization {
            StorageUtilization::Full(_) => 0,
            StorageUtilization::Slice { offset, size: _ } => offset,
        };
        slice.storage.utilization = StorageUtilization::Slice { offset, size };
        slice.padding = padding;

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

        self.chunks
            .insert(id, Chunk::new(storage, handle.clone(), Vec::new()));
        self.chunk_index.insert(id, size);

        handle
    }

    fn extend_max_memory<Storage: ComputeStorage>(&mut self, storage: &mut Storage, size: usize) {
        if self.debug {
            log::info!("Extend max memory ...");
        }

        let mut slices = Vec::<SliceUpdate>::new();
        let mut current_size = size;

        let chunks_sorted = self
            .chunk_index
            .find_by_size(0..self.max_chunk_size - 1)
            .map(Clone::clone)
            .collect::<Vec<_>>();

        let mut deallocations = HashSet::<ChunkId>::new();

        for chunk_id in chunks_sorted {
            let chunk = self.chunks.get(&chunk_id).unwrap();
            let chunk_id = *chunk.handle.id();
            let slices_ids = chunk.slices.clone();

            for slice_id in slices_ids {
                let slice = self.slices.get(&slice_id).unwrap();
                let size = slice.storage.size();

                let effective_size = slice.effective_size();
                current_size += effective_size;

                if current_size >= self.max_chunk_size {
                    let alloc_size = current_size - effective_size;
                    // let alloc_size = self.max_chunk_size;
                    self.move_to_new_chunk(alloc_size, storage, &mut slices, &mut deallocations);
                    current_size = effective_size;
                }

                slices.push(SliceUpdate { slice_id, size });
            }

            deallocations.insert(chunk_id);
        }

        if !slices.is_empty() {
            self.move_to_new_chunk(current_size, storage, &mut slices, &mut deallocations);
        } else {
            self.deallocate(storage, &mut deallocations);
        }
    }

    fn deallocate<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        deallocations: &mut HashSet<ChunkId>,
    ) {
        for id in deallocations.drain() {
            let mut chunk = self.chunks.remove(&id).unwrap();
            self.ring.remove_chunk(id);

            for slice in chunk.slices.drain(..) {
                let slice = self.slices.get(&slice).unwrap();
                let chunk_id = *slice.chunk.id();

                assert_ne!(chunk_id, id, "Chunk id should be updated");
            }

            self.chunk_index.remove(&id);
            storage.dealloc(chunk.storage.id);
        }
    }

    fn move_to_new_chunk<Storage: ComputeStorage>(
        &mut self,
        alloc_size: usize,
        storage: &mut Storage,
        slices: &mut Vec<SliceUpdate>,
        deallocations: &mut HashSet<ChunkId>,
    ) {
        let chunk = self.create_chunk(storage, alloc_size);
        let storage_id = self.chunks.get(chunk.id()).unwrap().storage.id.clone();
        let mut offset = 0;
        let mut slices_ids = Vec::new();

        for update in slices.drain(..) {
            let slice_id = update.slice_id;

            let slice = self.slices.get_mut(&slice_id).unwrap();
            let old_storage = slice.storage.clone();

            slice.chunk = chunk.clone();
            slice.storage = StorageHandle {
                id: storage_id.clone(),
                utilization: StorageUtilization::Slice {
                    offset,
                    size: update.size,
                },
            };
            storage.copy(&old_storage, &slice.storage);
            slices_ids.push(slice_id);

            offset += slice.effective_size();
        }

        let chunk = self.chunks.get_mut(chunk.id()).unwrap();
        chunk.slices = slices_ids;

        self.deallocate(storage, deallocations);
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

    fn split(&mut self, offset_slice: usize) -> Self {
        let size_new = self.effective_size() - offset_slice;
        let offset_new = self.storage.offset() + offset_slice;

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

        assert!(
            size_new >= BUFFER_ALIGNMENT,
            "Size new > {BUFFER_ALIGNMENT}"
        );
        let padding = calculate_padding(size_new - BUFFER_ALIGNMENT);

        Slice::new(storage_new, handle, self.chunk.clone(), padding)
    }

    fn id(&self) -> SliceId {
        *self.handle.id()
    }
}

impl MemoryChunk<Slice> for Chunk {
    fn merge_next_slice(
        &mut self,
        from_slice_index: usize,
        slices: &mut HashMap<SliceId, Slice>,
    ) -> bool {
        let slice_id_current = self.slices.get(from_slice_index).unwrap();
        let slice_id_next = self.slices.get(from_slice_index + 1);
        let slice_id_next = match slice_id_next {
            Some(val) => val,
            None => return false,
        };

        let slice_next = slices.get(slice_id_next).unwrap();
        let is_free = slice_next.is_free();
        let size = slice_next.effective_size();

        let slice_current = slices.get_mut(slice_id_current).unwrap();

        if is_free {
            slice_current.storage.utilization = StorageUtilization::Slice {
                size: slice_current.effective_size() + size,
                offset: slice_current.storage.offset(),
            };
            slices.remove(slice_id_next);
            self.slices.remove(from_slice_index + 1);

            return true;
        }

        false
    }

    fn slice(&self, index: usize) -> Option<SliceId> {
        self.slices.get(index).copied()
    }

    fn insert_slice(&mut self, position: usize, slice_id: SliceId) {
        self.slices.insert(position, slice_id);
    }
}
