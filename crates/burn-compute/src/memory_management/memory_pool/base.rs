use super::index::SearchIndex;
use super::{
    ChunkHandle, ChunkId, DroppedSlices, MemoryPoolBinding, MemoryPoolHandle, SliceHandle, SliceId,
};
use crate::storage::{ComputeStorage, StorageHandle, StorageUtilization};
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use hashbrown::{HashMap, HashSet};

pub struct MemoryPool {
    chunks: HashMap<ChunkId, Chunk>,
    slices: HashMap<SliceId, Slice>,
    merging_strategy: ChunkDefragmentationStrategy,
    rounding: RoundingStrategy,
    chunk_index: SearchIndex<ChunkId>,
    dropped: DroppedSlices,
    max_chunk_size: usize,
    ring: RingBuffer,
}

#[derive(new, Debug)]
struct Chunk {
    storage: StorageHandle,
    handle: ChunkHandle,
    slices: Vec<SliceId>,
}

impl Chunk {
    fn generate_mergings(
        &self,
        merging_map: &mut HashMap<ChunkId, Vec<Merging>>,
        slices: &mut HashMap<SliceId, Slice>,
        start_index: usize,
    ) {
        let mut to_merge: Vec<Merging> = Vec::new();

        let mut start_index: usize = 0;
        let mut num_merge = 0;
        let mut offset_current = 0;
        let mut offset = 0;
        let mut slices_ids = Vec::new();

        for (i, slice_id) in self.slices.iter().enumerate() {
            if i < start_index {
                continue;
            }

            let slice = slices.get(slice_id).unwrap();

            if slice.handle.is_free() {
                let effective_size = slice.effective_size();
                slices_ids.push(*slice_id);
                num_merge += 1;
                offset += effective_size;

                if i < self.slices.len() - 1 {
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
            merging_map.insert(*self.handle.id(), to_merge);
        }
    }

    fn merge_contiguous_slices(
        &mut self,
        mergings: &Vec<Merging>,
        slices: &mut HashMap<SliceId, Slice>,
    ) {
        let slice_ids = self.slices.clone();
        let mut slices_updated = Vec::new();

        let mut index = 0;

        for merging in mergings {
            let slice = self.create_slice(merging.offset, merging.size);
            let slice_id = *slice.handle.id();

            slices.insert(slice_id, slice);

            for i in index..merging.start {
                slices_updated.push(*slice_ids.get(i).unwrap());
            }
            index = merging.end + 1;
            slices_updated.push(slice_id);

            for slice_id_to_remove in merging.slice_ids.iter() {
                slices.remove(slice_id_to_remove);
            }
        }

        for i in index..slice_ids.len() {
            slices_updated.push(*slice_ids.get(i).unwrap());
        }
        self.slices = slices_updated;
    }

    fn create_slice(&self, offset: usize, size: usize) -> Slice {
        if offset > 0 && size < MIN_SIZE_NEEDED_TO_OFFSET {
            panic!("tried to create slice of size {size} with an offset while the size needs to atleast be of size {MIN_SIZE_NEEDED_TO_OFFSET} for offset support");
        }
        if offset % BUFFER_ALIGNMENT != 0 {
            panic!("slice with offset {offset} needs to be a multiple of {BUFFER_ALIGNMENT}");
        }
        let handle = SliceHandle::new();

        let storage = StorageHandle {
            id: self.storage.id.clone(),
            utilization: StorageUtilization::Slice { offset, size },
        };

        let padding = calculate_padding(size);

        Slice::new(storage, handle, self.handle.clone(), padding)
    }
}

#[derive(new, Debug)]
struct Slice {
    storage: StorageHandle,
    handle: SliceHandle,
    chunk: ChunkHandle,
    padding: usize,
}

#[derive(Debug)]
struct RingBuffer {
    ordered_chunks: BTreeMap<usize, ChunkId>,
    chunk_positions: HashMap<ChunkId, usize>,
    chunk_index: usize,
    cursor_slice: usize,
    cursor_chunk: usize,
    total: usize,
}

impl RingBuffer {
    fn new() -> Self {
        Self {
            ordered_chunks: BTreeMap::new(),
            chunk_positions: HashMap::new(),
            chunk_index: 0,
            cursor_slice: 0,
            cursor_chunk: 0,
            total: 0,
        }
    }

    fn push_chunk(&mut self, chunk_id: ChunkId) {
        self.ordered_chunks.insert(self.total, chunk_id);
        self.chunk_positions.insert(chunk_id, self.total);
        self.total += 1;
    }

    fn remove_chunk(&mut self, chunk_id: ChunkId) {
        let position = self
            .chunk_positions
            .get(&chunk_id)
            .expect("chunk should be in the chunk position BTree")
            .to_owned();

        self.ordered_chunks.remove(&position);
        self.chunk_positions.remove(&chunk_id);

        let keys_to_update: Vec<_> = self
            .ordered_chunks
            .range(position..)
            .map(|(&pos, _)| pos)
            .collect();

        for &old_pos in &keys_to_update {
            let new_pos = old_pos - 1;
            let id = self.ordered_chunks.remove(&old_pos).unwrap();
            self.ordered_chunks.insert(new_pos, id);
            self.chunk_positions.insert(id, new_pos);
        }

        self.total -= 1;
    }

    fn find_free_slice_in_chunk(
        &mut self,
        size: usize,
        chunk: &mut Chunk,
        slices: &mut HashMap<SliceId, Slice>,
    ) -> Option<SliceId> {
        let mut slice_index = self.cursor_slice;
        let mut merged = false;

        loop {
            let slice_id = if let Some(slice_id) = chunk.slices.get(slice_index) {
                slice_id
            } else {
                break;
            };

            let slice = slices.get(slice_id).unwrap();

            let is_big_enough = slice.effective_size() >= size;
            let is_free = slice.handle.is_free();

            if is_big_enough && is_free {
                self.cursor_slice = slice_index + 1;

                if self.cursor_slice >= chunk.slices.len() {
                    self.cursor_slice = 0;
                }

                return Some(*slice_id);
            }

            if merged {
                break;
            }

            if is_free {
                let mut chunk_to_merged_slice: HashMap<ChunkId, Vec<Merging>> = HashMap::new();
                chunk.generate_mergings(&mut chunk_to_merged_slice, slices, slice_index);

                for (_, mergings) in chunk_to_merged_slice.into_iter() {
                    chunk.merge_contiguous_slices(&mergings, slices);
                }

                merged = true;
            }
            slice_index += 1;
        }

        self.cursor_slice = 0;
        None
    }

    fn find_free_slice(
        &mut self,
        size: usize,
        chunks: &mut HashMap<ChunkId, Chunk>,
        slices: &mut HashMap<SliceId, Slice>,
        max_cursor_position: usize,
    ) -> Option<SliceId> {
        let start = self.cursor_chunk;
        let end = usize::min(self.total, max_cursor_position);

        for chunk_index in start..end {
            if let Some(id) = self.ordered_chunks.get(&chunk_index) {
                let chunk = chunks.get_mut(id).unwrap();

                let result = self.find_free_slice_in_chunk(size, chunk, slices);

                if result.is_some() {
                    self.cursor_chunk = chunk_index;
                    return result;
                }
            }
            self.cursor_chunk = chunk_index;
        }

        self.cursor_chunk = 0;

        None
    }

    fn find_free_slice_two_ways(
        &mut self,
        size: usize,
        chunks: &mut HashMap<ChunkId, Chunk>,
        slices: &mut HashMap<SliceId, Slice>,
    ) -> Option<SliceId> {
        let max_second = self.cursor_chunk;
        let result = self.find_free_slice(size, chunks, slices, self.total);

        if result.is_some() {
            return result;
        }

        self.find_free_slice(size, chunks, slices, max_second)
    }
}

//impl ChunkRingBuffer {
//    fn new() -> Self {
//        Self {
//            // needs to be updated if we change chunks in the memory pool
//            last_chunk_position: None,
//            last_slice_offset: None,
//            chunks: Vec::new(),
//        }
//    }
//
//    fn get_next_fit(
//        &mut self,
//        chunks: &HashMap<ChunkId, Chunk>,
//        slices: &HashMap<SliceId, Slice>,
//        alloc_size: usize,
//    ) -> Option<SliceId> {
//        let last_chunk_id = match &self.last_chunk_position {
//            Some(chunk_position) => Some(self.chunks.get(*chunk_position).unwrap()),
//            None => {
//                let chunk = self.chunks.get(0);
//                if chunk.is_some() {
//                    self.last_chunk_position = Some(0);
//                    self.last_slice_offset = None;
//                }
//                chunk
//            }
//        };
//        let last_chunk = chunks.get(last_chunk_id?);
//        let mut last_chunk = match last_chunk {
//            Some(chunk) => {
//                let current_slice_id_offset_and_positon =
//                    chunk.get_next_slice(self.last_slice_offset, slices);
//                while current_slice_id_offset_and_positon.is_some() {
//                    let current_slice_id = current_slice_id_offset_and_positon.unwrap().slice_id;
//                    let current_slice = slices.get(&current_slice_id).unwrap();
//                    let alloc_can_fit = current_slice.effective_size() >= alloc_size
//                        && current_slice.handle.is_free();
//                    if alloc_can_fit {
//                        return Some(current_slice_id);
//                    }
//                    let current_slice_position =
//                        current_slice_id_offset_and_positon.unwrap().position;
//                    let current_slice_id_offset_and_positon =
//                        chunk.get_specific_slice(current_slice_position + 1);
//                }
//            }
//            // no chunks at all
//            None => {
//                return None;
//            }
//        };
//    }
//}
//
//#[derive(Debug, new)]
//struct SliceIdOffsetPosition {
//    slice_id: SliceId,
//    offset: usize,
//    position: usize,
//}
//
//impl Chunk {
//    fn get_next_slice(
//        &self,
//        last_offset: Option<usize>,
//        slices: &HashMap<SliceId, Slice>,
//    ) -> Option<SliceIdOffsetPosition> {
//        match last_offset {
//            Some(last_offset) => {
//                let mut temp_offset = 0;
//                for i in 0..(self.slices.len() - 1) {
//                    let slice_id = self.slices.get(i).unwrap();
//                    let slice = slices.get(slice_id).unwrap();
//                    temp_offset += slice.effective_size();
//                    if temp_offset > last_offset {
//                        return Some(SliceIdOffsetPosition::new(
//                            *self.slices.get(i + 1).unwrap(),
//                            temp_offset,
//                            i + 1,
//                        ));
//                    }
//                }
//                return None;
//            }
//            // assumes slices are stored contiguously in the chunk's vec
//            None => {
//                return Some(SliceIdOffsetPosition::new(
//                    *self.slices.get(0).expect("chunk shouldn't have 0 slices"),
//                    0,
//                    0,
//                ));
//            }
//        }
//    }
//
//    fn get_specific_slice(&self, position: usize) -> Option<SliceId> {
//        return self.slices.get(position).copied();
//    }
//}

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
            chunk_index: SearchIndex::new(),
            dropped: Arc::new(spin::Mutex::new(Vec::new())),
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
        log::info!("the memory usage is {ratio}");
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

                slices.push(first_slice_id);
                slices.push(slice_end_id);
            }
        }

        self.slices.remove(slice_to_split_id);

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
        let mut slice_id_size_pair: Option<(SliceId, usize)> = None;

        let slice_id =
            self.ring
                .find_free_slice_two_ways(effective_size, &mut self.chunks, &mut self.slices);
        if let Some(slice_id) = slice_id {
            let slice = self.slices.get(&slice_id).unwrap();
            slice_id_size_pair = Some((slice_id, slice.effective_size()));
        }

        if slice_id_size_pair.is_none() {
            return None;
        }
        let slice_id = slice_id_size_pair.unwrap().0;
        let slice_size = slice_id_size_pair.unwrap().1;

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

            return Some(slice.handle.clone());
        }

        if size < MIN_SIZE_NEEDED_TO_OFFSET {
            return None;
        }

        assert_eq!(size_diff % BUFFER_ALIGNMENT, 0);

        // split into 2 if needed
        let handle = self
            .split_slice_in_two(&slice_id, size)
            .expect("split should have returned a handle");

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

        self.ring.push_chunk(id);

        //assert_eq!(start_size % BUFFER_ALIGNMENT, 0);
        //assert_eq!(end_size % BUFFER_ALIGNMENT, 0);

        self.chunks
            .insert(id, Chunk::new(storage, handle.clone(), Vec::new()));
        self.chunk_index.insert(id, size);

        handle
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
