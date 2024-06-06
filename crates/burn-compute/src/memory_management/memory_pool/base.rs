use super::index::SearchIndex;
use super::{
    ChunkHandle, ChunkId, DroppedSlices, MemoryPoolBinding, MemoryPoolHandle, SliceHandle, SliceId,
};
use crate::storage::{ComputeStorage, StorageHandle, StorageUtilization};
use alloc::sync::Arc;
use hashbrown::{HashMap, HashSet};

pub struct MemoryPool {
    chunks: HashMap<ChunkId, Chunk>,
    slices: HashMap<SliceId, Slice>,
    merging_strategy: ChunkDefragmentationStrategy,
    rounding: RoundingStrategy,
    slice_size_index: SearchIndex<SliceId>,
    chunk_index: SearchIndex<ChunkId>,
    dropped: DroppedSlices,
    max_chunk_size: usize,
}

#[derive(new, Debug)]
struct Chunk {
    storage: StorageHandle,
    handle: ChunkHandle,
    slices: Vec<SliceId>,
}

#[derive(new, Debug)]
struct Slice {
    storage: StorageHandle,
    handle: SliceHandle,
    chunk: ChunkHandle,
    padding: usize,
}

impl Slice {
    pub fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }
}

const MIN_SIZE_NEEDED_TO_OFFSET: usize = 16;
const BUFFER_ALIGNMENT: usize = 32;
const CHUNK_ROUNDING: usize = 2 * 1024 * 1024; // 2 MB

pub enum RoundingStrategy {
    RoundUp,
    None,
}

impl RoundingStrategy {
    fn alloc_size(&self, size: usize) -> usize {
        match self {
            RoundingStrategy::RoundUp => {
                if size < CHUNK_ROUNDING {
                    return size;
                }

                let factor = (size + (CHUNK_ROUNDING - 1)) / CHUNK_ROUNDING;

                factor * CHUNK_ROUNDING
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
struct Merging {
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
    ) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            merging_strategy,
            rounding: alloc_strategy,
            max_chunk_size,
            slice_size_index: SearchIndex::new(),
            chunk_index: SearchIndex::new(),
            dropped: Arc::new(spin::Mutex::new(Vec::new())),
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
        self.merge_unused_slices();
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
            self.chunk_defragmentation(storage);

            if let Some(handle) = self.get_free_slice(size) {
                return MemoryPoolHandle {
                    slice: handle,
                    dropped: self.dropped.clone(),
                };
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
            dropped: self.dropped.clone(),
        };

        let extra_slice = if effective_size < alloc_size {
            Some(self.create_slice(effective_size, alloc_size - effective_size, handle_chunk))
        } else {
            None
        };

        // Update chunk metadata.
        let chunk = self.chunks.get_mut(&chunk_id).unwrap();

        self.slice_size_index.insert(slice_id, effective_size);

        self.slices.insert(slice_id, slice);
        chunk.slices.push(slice_id);

        if let Some(slice) = extra_slice {
            let id = *slice.handle.id();
            self.slice_size_index.insert(id, slice.effective_size());

            self.slices.insert(id, slice);
            chunk.slices.push(id);
        }

        returned
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
                dropped: self.dropped.clone(),
            },
            None => {
                let handle = self.alloc(storage, size, sync);
                self.slice_size_index.remove(handle.slice.id());
                handle
            }
        }
    }

    /// Tries to split a slice in two with the first slice being of the specified size
    /// If there is not enough space for 2 slice, only uses one slice
    /// returns the handle of the first slice
    fn split_slice_in_two(
        &mut self,
        slice_to_split_id: &SliceId,
        first_slice_size: usize,
    ) -> Option<SliceHandle> {
        let slice_to_split = self.slices.get(slice_to_split_id).unwrap();
        let slice_to_split_effective_size = slice_to_split.effective_size();
        let chunk = if let Some(chunk) = self.chunks.get_mut(slice_to_split.chunk.id()) {
            chunk
        } else {
            panic!("Can't use chunk {:?}", slice_to_split.chunk.id());
        };
        let current_slice_chunk_handle = chunk.handle.clone();

        let mut slices = Vec::with_capacity(chunk.slices.len() + 1);
        let mut offset = 0;

        let mut slices_old = Vec::new();
        core::mem::swap(&mut slices_old, &mut chunk.slices);

        let mut handle = None;
        for slice_id in slices_old.into_iter() {
            // Assumes that all slices are contiguous in a chunk.
            let slice = self.slices.get(&slice_id).unwrap();

            if slice_id != *slice_to_split_id {
                slices.push(slice_id);
                offset += slice.effective_size();
            } else {
                let first_slice =
                    self.create_slice(offset, first_slice_size, current_slice_chunk_handle.clone());
                let first_slice_id = *first_slice.handle.id();
                let first_slice_effective_size = first_slice.effective_size();
                offset += first_slice_effective_size;

                let second_slice_size = slice_to_split_effective_size - first_slice_effective_size;
                let slice_end = self.create_slice(
                    offset,
                    second_slice_size,
                    current_slice_chunk_handle.clone(),
                );
                let slice_end_id = *slice_end.handle.id();
                let slice_end_effective_size = slice_end.effective_size();
                offset += slice_end_effective_size;

                let created_offset = first_slice_effective_size + slice_end_effective_size;
                assert_eq!(created_offset, slice_to_split_effective_size);

                handle = Some(first_slice.handle.clone());
                self.slices.insert(first_slice_id, first_slice);
                self.slices.insert(slice_end_id, slice_end);

                self.slice_size_index
                    .insert(first_slice_id, first_slice_effective_size);
                self.slice_size_index
                    .insert(slice_end_id, slice_end_effective_size);

                slices.push(first_slice_id);
                slices.push(slice_end_id);
            }
        }

        self.slices.remove(slice_to_split_id);
        self.slice_size_index.remove(slice_to_split_id);

        let chunk = self
            .chunks
            .get_mut(current_slice_chunk_handle.id())
            .unwrap();
        chunk.slices = slices;
        handle
    }

    /// Finds the smallest of the free and large enough chunks to fit `size`
    /// Returns the chunk's id and size.
    fn get_free_slice(&mut self, size: usize) -> Option<SliceHandle> {
        let padding = Self::calculate_padding(size);
        let effective_size = size + padding;

        let (slice_id, slice_size) = match self
            .slice_size_index
            .find_by_size(effective_size..usize::MAX)
            .next()
        {
            Some(id) => {
                let slice = self.slices.get(id).unwrap();

                (*id, slice.effective_size())
            }
            None => return None,
        };

        let size_diff = slice_size - effective_size;

        // if same size reuse the slice
        if size_diff == 0 {
            let slice = self.slices.get_mut(&slice_id).unwrap();

            let offset = match slice.storage.utilization {
                StorageUtilization::Full(_) => 0,
                StorageUtilization::Slice { offset, size: _ } => offset,
            };
            slice.storage.utilization = StorageUtilization::Slice { offset, size };
            slice.padding = padding;

            self.slice_size_index.remove(slice.handle.id());
            return Some(slice.handle.clone());
        }

        assert_eq!(size_diff % BUFFER_ALIGNMENT, 0);

        // split into 2 if needed
        let handle = self
            .split_slice_in_two(&slice_id, effective_size)
            .expect("split should have returned a handle");

        self.slice_size_index.remove(handle.id());
        return Some(handle);
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
        let size = storage.size();

        self.chunks
            .insert(id, Chunk::new(storage, handle.clone(), Vec::new()));
        self.chunk_index.insert(id, size);

        handle
    }

    // generates and adds all the merging information of a given chunk to an hash_map
    fn generate_mergings(&self, chunk: &Chunk, merging_map: &mut HashMap<ChunkId, Vec<Merging>>) {
        let mut to_merge: Vec<Merging> = Vec::new();

        let mut start_index: usize = 0;
        let mut num_merge = 0;
        let mut offset_current = 0;
        let mut offset = 0;
        let mut slices_ids = Vec::new();

        for (i, slice_id) in chunk.slices.iter().enumerate() {
            let slice = self.slices.get(slice_id).unwrap();

            if slice.handle.is_free() {
                let effective_size = slice.effective_size();
                slices_ids.push(*slice_id);
                num_merge += 1;
                offset += effective_size;

                if i < chunk.slices.len() - 1 {
                    continue;
                }
            }

            if num_merge > 1 {
                let mut empty = Vec::new();
                core::mem::swap(&mut slices_ids, &mut empty);
                let merging = Merging {
                    start: start_index,
                    end: start_index + num_merge - 1,
                    offset: offset_current,
                    size: offset - offset_current,
                    slice_ids: empty,
                };
                to_merge.push(merging);
            }

            offset += slice.effective_size();
            start_index = i + 1;
            num_merge = 0;
            offset_current = offset;
            slices_ids.clear();
        }
        if !to_merge.is_empty() {
            merging_map.insert(*chunk.handle.id(), to_merge);
        }
    }

    // merges all free slices together use the mergings metadata
    fn merge_contiguous_slices(&mut self, chunk_id: ChunkId, mergings: &Vec<Merging>) {
        let chunk = self.chunks.get(&chunk_id).unwrap();
        let chunk_handle = chunk.handle.clone();
        let slices = chunk.slices.clone();
        let mut slices_updated = Vec::new();

        let mut index = 0;

        for merging in mergings {
            let slice = self.create_slice(merging.offset, merging.size, chunk_handle.clone());
            let slice_id = *slice.handle.id();
            let effective_size = slice.effective_size();

            self.slice_size_index.insert(slice_id, effective_size);
            self.slices.insert(slice_id, slice);

            for i in index..merging.start {
                slices_updated.push(*slices.get(i).unwrap());
            }
            index = merging.end + 1;
            slices_updated.push(slice_id);

            for slice_id_to_remove in merging.slice_ids.iter() {
                self.slice_size_index.remove(slice_id_to_remove);
                self.slices.remove(slice_id_to_remove);
            }
        }

        for i in index..slices.len() {
            slices_updated.push(*slices.get(i).unwrap());
        }
        let chunk = self.chunks.get_mut(&chunk_id).unwrap();
        core::mem::swap(&mut chunk.slices, &mut slices_updated);
    }

    // Merge all contiguous free_slices together, assumes that slices are in sorted order.
    fn merge_unused_slices(&mut self) {
        let mut chunks = HashSet::new();
        for dropped in self.dropped.lock().drain(..) {
            if let Some(slice) = self.slices.get(&dropped) {
                self.slice_size_index
                    .insert(dropped, slice.effective_size());

                chunks.insert(*slice.chunk.id());
            }
        }

        let mut chunk_to_merged_slice: HashMap<ChunkId, Vec<Merging>> = HashMap::new();

        for id in chunks {
            let chunk = self.chunks.get(&id).unwrap();
            self.generate_mergings(chunk, &mut chunk_to_merged_slice);
        }

        for (chunk_id, mergings) in chunk_to_merged_slice.into_iter() {
            self.merge_contiguous_slices(chunk_id, &mergings);
        }
    }

    fn chunk_defragmentation<Storage: ComputeStorage>(&mut self, storage: &mut Storage) {
        log::info!("Chunk defragmentation ...");

        let mut slices = Vec::<SliceUpdate>::new();
        let mut current_size = 0;

        let chunks_sorted = self
            .chunk_index
            .find_by_size(0..self.max_chunk_size)
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
