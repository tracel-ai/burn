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
    merging_strategy: ChunkDefragmentationStrategy,
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
    // TODO : Figure out a better name for this
    RoundUp,
    None,
}

impl RoundingStrategy {
    fn alloc_size(&self, size: usize) -> usize {
        match self {
            RoundingStrategy::RoundUp => {
                if size < 32 {
                    return 32;
                }

                if size < MB {
                    return 2 * MB;
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
pub enum ChunkDefragmentationStrategy {
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

impl ChunkDefragmentationStrategy {
    /// Create a new strategy with the given period.
    pub fn new_period_tick(period: usize) -> Self {
        ChunkDefragmentationStrategy::PeriodTick { period, state: 0 }
    }

    fn should_perform_defragmentation(&mut self) -> bool {
        match self {
            ChunkDefragmentationStrategy::PeriodTick { period, state } => {
                *state = (*state + 1) % *period;
                *state == 0
            }
            ChunkDefragmentationStrategy::Never => false,
        }
    }
}

struct SliceUpdate {
    slice_id: SliceId,
    size: usize,
}

#[derive(Debug)]
pub struct Merging {
    start: usize,
    end: usize,
    offset: usize,
    size: usize,
    slice_ids: Vec<SliceId>,
}

impl MemoryPool {
    pub fn new(
        merging_strategy: ChunkDefragmentationStrategy,
        alloc_strategy: RoundingStrategy,
        max_chunk_size: usize,
        debug: bool,
    ) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            merging_strategy,
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
        self.display_memory_usage();
        self.reserve_algorithm(storage, size, sync)
    }

    pub fn alloc<Storage: ComputeStorage, Sync: FnOnce()>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        sync: Sync,
    ) -> MemoryPoolHandle {
        let may_perform_full_defrag = self.merging_strategy.should_perform_defragmentation();

        if may_perform_full_defrag {
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

        let slice = self.create_slice(0, slice_size, handle_chunk.clone());
        let effective_size = slice.effective_size();

        let handle_slice = slice.handle.clone();
        let slice_id = *handle_slice.id();
        let returned = MemoryPoolHandle {
            slice: handle_slice,
        };

        let extra_slice = if effective_size < alloc_size {
            Some(self.create_slice(effective_size, alloc_size - effective_size, handle_chunk))
        } else {
            None
        };

        // Update chunk metadata.
        let chunk = self.chunks.get_mut(&chunk_id).unwrap();

        self.slices.insert(slice_id, slice);
        chunk.slices.push(slice_id);

        if let Some(slice) = extra_slice {
            let id = *slice.handle.id();

            self.slices.insert(id, slice);
            chunk.slices.push(id);
        }

        returned
    }

    fn display_memory_usage(&self) {
        let mut total_memory_usage: f64 = 0.0;
        for (.., chunk) in self.chunks.iter() {
            total_memory_usage += chunk.storage.size() as f64;
        }
        let mut effective_memory_usage: f64 = 0.0;
        for (.., slice) in self.slices.iter() {
            if slice.handle.is_free() {
                effective_memory_usage += slice.storage.size() as f64;
            }
        }
        let ratio = 100.0 * effective_memory_usage / total_memory_usage;
        // log::info!("the memory usage is {ratio}");
    }

    fn reserve_algorithm<Storage: ComputeStorage, Sync: FnOnce()>(
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

    /// Finds the smallest of the free and large enough chunks to fit `size`
    /// Returns the chunk's id and size.
    fn get_free_slice(&mut self, size: usize) -> Option<SliceHandle> {
        if size < MIN_SIZE_NEEDED_TO_OFFSET {
            return None;
        }

        let padding = Self::calculate_padding(size);
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
        if offset > 0 && size < MIN_SIZE_NEEDED_TO_OFFSET {
            panic!("tried to create slice of size {size} with an offset while the size needs to atleast be of size {MIN_SIZE_NEEDED_TO_OFFSET} for offset support");
        }
        if offset % BUFFER_ALIGNMENT != 0 {
            panic!("slice with offset {offset} needs to be a multiple of {BUFFER_ALIGNMENT}");
        }
        let chunk = self.chunks.get(handle_chunk.id()).unwrap();
        let handle = SliceHandle::new();

        let storage = StorageHandle {
            id: chunk.storage.id.clone(),
            utilization: StorageUtilization::Slice { offset, size },
        };

        let padding = Self::calculate_padding(size);

        Slice::new(storage, handle, chunk.handle.clone(), padding)
    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
    ) -> ChunkHandle {
        let padding = Self::calculate_padding(size);
        let effective_size = size + padding;

        let storage = storage.alloc(effective_size);
        let handle = ChunkHandle::new();
        let id = *handle.id();

        self.ring.push_chunk(id);

        //assert_eq!(start_size % BUFFER_ALIGNMENT, 0);
        //assert_eq!(end_size % BUFFER_ALIGNMENT, 0);

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
            let chunk_id = chunk.handle.id().clone();
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

    fn calculate_padding(size: usize) -> usize {
        let rem = size % BUFFER_ALIGNMENT;
        if rem != 0 {
            BUFFER_ALIGNMENT - rem
        } else {
            0
        }
    }
}

fn calculate_padding(size: usize) -> usize {
    let rem = size % BUFFER_ALIGNMENT;
    if rem != 0 {
        BUFFER_ALIGNMENT - rem
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

        assert!(size_new >= 32, "Size new > 32");
        let padding = calculate_padding(size_new - 32);

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
        // let mut chunk_to_merged_slice: HashMap<ChunkId, Vec<Merging>> = HashMap::new();
        // self.generate_mergings(&mut chunk_to_merged_slice, slices, from_slice_index);
        // for (_, mergings) in chunk_to_merged_slice.into_iter() {
        //     self.merge_contiguous_slices(&mergings, slices);
        // }
        // log::info!("Num slices {}", self.slices.len());
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

        // for (index, slice_id) in self.slices.drain(..).enumerate() {
        //     if index < from_slice_index {
        //         slices_updated.push(slice_id);
        //         continue;
        //     }

        //     let slice = slices.remove(&slice_id).unwrap();

        //     if slice.is_free() {
        //         match current.take() {
        //             Some(mut val) => {
        //                 val.storage.utilization = StorageUtilization::Slice {
        //                     size: val.effective_size() + slice.effective_size(),
        //                     offset: val.storage.offset(),
        //                 };
        //                 current = Some(val);
        //             }
        //             None => {
        //                 current = Some(slice);
        //             }
        //         };
        //     } else {
        //         if let Some(s) = current.take() {
        //             slices_updated.push(s.id());
        //             slices.insert(s.id(), s);
        //         }

        //         slices_updated.push(slice.id());
        //         slices.insert(slice.id(), slice);
        //     }
        // }

        // if let Some(s) = current.take() {
        //     slices_updated.push(s.id());
        //     slices.insert(s.id(), s);
        // }

        // self.slices = slices_updated;
    }

    fn slice(&self, index: usize) -> Option<SliceId> {
        self.slices.get(index).map(Clone::clone)
    }

    fn insert_slice(&mut self, position: usize, slice_id: SliceId) {
        self.slices.insert(position, slice_id);
    }
}
