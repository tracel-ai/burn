use crate::{
    memory_id_type,
    storage::{ComputeStorage, StorageHandle, StorageUtilization},
};
use alloc::vec::Vec;
use hashbrown::HashMap;

#[cfg(all(not(target_family = "wasm"), feature = "std"))]
use std::time;
#[cfg(all(target_family = "wasm", feature = "std"))]
use web_time as time;

use super::{MemoryBinding, MemoryHandle, MemoryManagement};

// The ChunkId allows to keep track of how many references there are to a specific chunk.
memory_id_type!(ChunkId, ChunkHandle, ChunkBinding);
// The SliceId allows to keep track of how many references there are to a specific slice.
memory_id_type!(SliceId, SliceHandle, SliceBinding);

/// A tensor memory handle, referring to either a chunk or a slice.
#[derive(Debug, Clone)]
pub enum DynamicHandle {
    /// A whole chunk of memory.
    Chunk(ChunkHandle),
    /// A slice of a chunk of memory.
    Slice(SliceHandle),
}

/// Binding of the [dynamic handle](DynamicHandle).
#[derive(Debug, Clone)]
pub enum DynamicBinding {
    /// Binding of the [chunk handle](ChunkHandle).
    Chunk(ChunkBinding),
    /// Binding of the [slice handle](SliceHandle)
    Slice(SliceBinding),
}

/// The strategy defines the frequency at which merging of free slices (defragmentation) occurs
#[derive(Debug)]
pub enum MergingStrategy {
    /// Once every n calls to reserve.
    PeriodTick {
        /// Number of calls to be executed before triggering the defragmentation.
        period: usize,
        /// Current state. Should start at zero.
        state: usize,
    },
    #[cfg(feature = "std")]
    /// Once every period of time
    PeriodTime {
        /// Number of time before triggering the defragmentation.
        period: time::Duration,
        /// Current state. Should start at now.
        state: time::Instant,
    },
    /// Never defragment.
    Never,
}

/// The strategy defines when to reuse chunk with slices.
#[derive(Debug)]
pub enum SliceStrategy {
    /// Never use slices.
    Never,
    /// Ratio needed before the chunk can be used as a slice. Between 0 and 1.
    Ratio(f32),
    /// When the reserved memory is at least {} bytes.
    MinimumSize(usize),
    /// When the reserved memory less than {} bytes.
    MaximumSize(usize),
}

impl SliceStrategy {
    /// If the chunk can be used with a slice.
    pub fn can_use_chunk(&self, chunk_size: usize, reserved_size: usize) -> bool {
        if chunk_size < reserved_size {
            return false;
        }

        match self {
            SliceStrategy::Never => false,
            SliceStrategy::Ratio(ratio) => (reserved_size as f32 / chunk_size as f32) >= *ratio,
            SliceStrategy::MinimumSize(bytes) => reserved_size >= *bytes,
            SliceStrategy::MaximumSize(bytes) => reserved_size <= *bytes,
        }
    }
}

impl MergingStrategy {
    /// Create a new strategy with the given period.
    pub fn new_period_tick(period: usize) -> Self {
        MergingStrategy::PeriodTick { period, state: 0 }
    }

    fn should_perform_defragmentation(&mut self) -> bool {
        match self {
            MergingStrategy::PeriodTick { period, state } => {
                *state = (*state + 1) % *period;
                *state == 0
            }
            #[cfg(feature = "std")]
            MergingStrategy::PeriodTime { period, state } => {
                if &state.elapsed() > period {
                    *state = time::Instant::now();
                    true
                } else {
                    false
                }
            }
            MergingStrategy::Never => false,
        }
    }
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

#[derive(Debug)]
struct Merging {
    start: usize,
    end: usize,
    offset: usize,
    size: usize,
    slice_ids: Vec<SliceId>,
}

impl Slice {
    pub fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }
}

const MIN_SIZE_NEEDED_TO_OFFSET: usize = 16;
const BUFFER_ALIGNMENT: usize = 32;

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct DynamicMemoryManagement<Storage> {
    chunks: HashMap<ChunkId, Chunk>,
    slices: HashMap<SliceId, Slice>,
    merging_strategy: MergingStrategy,
    slice_strategy: SliceStrategy,
    storage: Storage,
}

impl<Storage> core::fmt::Debug for DynamicMemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "DynamicMemoryManagement {:?} - {:?}",
                self.merging_strategy,
                core::any::type_name::<Storage>(),
            )
            .as_str(),
        )
    }
}

impl MemoryBinding for DynamicBinding {}

impl MemoryHandle<DynamicBinding> for DynamicHandle {
    fn can_mut(&self) -> bool {
        match &self {
            DynamicHandle::Chunk(id) => id.can_mut(),
            DynamicHandle::Slice(id) => id.can_mut(),
        }
    }

    fn binding(self) -> DynamicBinding {
        match self {
            Self::Chunk(handle) => DynamicBinding::Chunk(handle.binding()),
            Self::Slice(handle) => DynamicBinding::Slice(handle.binding()),
        }
    }
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> for DynamicMemoryManagement<Storage> {
    type Handle = DynamicHandle;
    type Binding = DynamicBinding;

    /// Returns the resource from the storage, for the specified handle.
    fn get(&mut self, binding: Self::Binding) -> Storage::Resource {
        let storage = match binding {
            DynamicBinding::Chunk(chunk) => {
                &self
                    .chunks
                    .get(chunk.id())
                    .expect("Storage found for the given execution buffer handle")
                    .storage
            }
            DynamicBinding::Slice(slice) => {
                &self
                    .slices
                    .get(slice.id())
                    .expect("Storage found for the given execution buffer handle")
                    .storage
            }
        };

        self.storage.get(storage)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    fn reserve(&mut self, size: usize) -> Self::Handle {
        let handle = self.reserve_algorithm(size);

        if self.merging_strategy.should_perform_defragmentation() {
            self.defragmentation();
        }

        handle
    }

    fn alloc(&mut self, size: usize) -> Self::Handle {
        let handle_chunk = self.create_chunk(size);
        let chunk_id = *handle_chunk.id();

        let slice = self.create_slice(0, size, handle_chunk);
        let handle_slice = slice.handle.clone();

        let chunk = self.chunks.get_mut(&chunk_id).unwrap();
        chunk.slices.push(*handle_slice.id());

        self.slices.insert(*handle_slice.id(), slice);

        DynamicHandle::Slice(handle_slice)
    }

    fn dealloc(&mut self, binding: Self::Binding) {
        match binding {
            DynamicBinding::Chunk(chunk) => {
                if let Some(chunk) = self.chunks.remove(chunk.id()) {
                    self.storage.dealloc(chunk.storage.id);
                }
            }
            DynamicBinding::Slice(_) => panic!("Can't dealloc slice manually"),
        }
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }
}

impl<Storage: ComputeStorage> DynamicMemoryManagement<Storage> {
    /// Creates a new instance using the given storage, merging_strategy strategy and slice strategy.
    pub fn new(
        storage: Storage,
        merging_strategy: MergingStrategy,
        slice_strategy: SliceStrategy,
    ) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            merging_strategy,
            slice_strategy,
            storage,
        }
    }

    fn reserve_algorithm(&mut self, size: usize) -> DynamicHandle {
        // Looks for a large enough, existing but unused chunk of memory.
        let slice = self.get_free_slice(size);

        match slice {
            Some(slice) => DynamicHandle::Slice(slice.clone()),
            None => self.alloc(size),
        }
    }

    fn find_free_slice_best_fit(
        &self,
        size: usize,
        effective_size: usize,
    ) -> Option<(SliceId, usize)> {
        let mut size_diff_current = usize::MAX;
        let mut found = None;
        for (__, chunk) in self.chunks.iter() {
            if size < MIN_SIZE_NEEDED_TO_OFFSET && chunk.slices.len() > 1 {
                continue;
            }
            if !self
                .slice_strategy
                .can_use_chunk(chunk.storage.size(), effective_size)
            {
                continue;
            }
            for slice_id in chunk.slices.iter() {
                let slice = self.slices.get(slice_id).unwrap();
                let slice_can_be_reused =
                    slice.handle.is_free() && slice.effective_size() >= effective_size;

                if slice_can_be_reused {
                    let size_diff = slice.effective_size() - effective_size;
                    if size_diff < size_diff_current {
                        size_diff_current = size_diff;
                        found = Some((*slice_id, size_diff));
                    }
                }
            }
        }
        found
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
        let chunk = self.chunks.get_mut(slice_to_split.chunk.id()).unwrap();
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
                offset += first_slice.effective_size();

                let second_slice_size =
                    slice_to_split_effective_size - first_slice.effective_size();
                let slice_end = self.create_slice(
                    offset,
                    second_slice_size,
                    current_slice_chunk_handle.clone(),
                );
                let slice_end_id = *slice_end.handle.id();
                offset += slice_end.effective_size();

                let created_offset = first_slice.effective_size() + slice_end.effective_size();
                assert_eq!(created_offset, slice.effective_size());

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

        let found = self.find_free_slice_best_fit(size, effective_size);
        let (slice_id, size_diff_current) = match found {
            Some(val) => val,
            None => {
                return None;
            }
        };

        // if same size reuse the slice
        if size_diff_current == 0 {
            let slice = self.slices.get_mut(&slice_id).unwrap();
            let offset = match slice.storage.utilization {
                StorageUtilization::Full(_) => 0,
                StorageUtilization::Slice { offset, size: _ } => offset,
            };
            slice.storage.utilization = StorageUtilization::Slice { offset, size };
            slice.padding = padding;

            return Some(self.slices.get(&slice_id).unwrap().handle.clone());
        }

        assert_eq!(size_diff_current % BUFFER_ALIGNMENT, 0);

        // split into 2 if needed
        let handle = self.split_slice_in_two(&slice_id, size);
        if handle.is_none() {
            panic!("split should have returned a handle");
        }

        handle
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

    #[cfg(test)]
    fn insert_slice(&mut self, slice: Slice, chunk_id: ChunkId) {
        let slice_id = *slice.handle.id();
        self.slices.insert(*slice.handle.id(), slice);
        let chunk = self.chunks.get_mut(&chunk_id).unwrap();
        chunk.slices.push(slice_id);
    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk(&mut self, size: usize) -> ChunkHandle {
        let padding = Self::calculate_padding(size);
        let effective_size = size + padding;

        let storage = self.storage.alloc(effective_size);
        let handle = ChunkHandle::new();

        self.chunks.insert(
            *handle.id(),
            Chunk::new(storage, handle.clone(), Vec::new()),
        );

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
                slices_ids.push(*slice_id);
                num_merge += 1;
                offset += slice.effective_size();
                continue;
            } else if num_merge > 1 {
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
            self.slices.insert(slice_id, slice);
            for i in index..merging.start {
                slices_updated.push(*slices.get(i).unwrap());
            }
            index = merging.end + 1;
            slices_updated.push(slice_id);

            for slice_id_to_remove in merging.slice_ids.iter() {
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
    fn defragmentation(&mut self) {
        let mut chunk_to_merged_slice: HashMap<ChunkId, Vec<Merging>> = HashMap::new();
        for (.., chunk) in self.chunks.iter() {
            self.generate_mergings(chunk, &mut chunk_to_merged_slice);
        }

        for (chunk_id, mergings) in chunk_to_merged_slice.into_iter() {
            self.merge_contiguous_slices(chunk_id, &mergings);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        memory_management::{MemoryHandle, MemoryManagement},
        storage::BytesStorage,
    };

    #[test]
    fn can_mut_with_single_tensor_reference() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::Never,
            SliceStrategy::Never,
        );

        let chunk_size = 4;
        let simple_handle = memory_management.create_chunk(chunk_size);

        let x = simple_handle.clone();
        core::mem::drop(simple_handle);

        assert!(x.can_mut());
    }

    #[test]
    fn two_tensor_references_remove_mutability() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::Never,
            SliceStrategy::Never,
        );

        let chunk_size = 4;
        let simple_handle = memory_management.create_chunk(chunk_size);

        let x = simple_handle.clone();

        assert!(!simple_handle.can_mut());
        assert!(!x.can_mut())
    }

    #[test]
    fn when_non_empty_chunk_exists_and_other_one_created_there_should_be_two() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::Never,
            SliceStrategy::Never,
        );
        let chunk_size = 4;
        let _chunk_handle = memory_management.reserve(chunk_size);
        let _new_handle = memory_management.reserve(chunk_size);

        assert_eq!(memory_management.chunks.len(), 2);
    }

    #[test]
    fn when_big_chunk_is_freed_should_be_filled_with_smaller_slices() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::Never,
            SliceStrategy::Ratio(0.2),
        );
        let big_slice_size = 32 * 3;
        let small_slice_size = 32;

        let big_slice = memory_management.reserve(big_slice_size);
        drop(big_slice);
        let _small_slice_1 = memory_management.reserve(small_slice_size);
        let _small_slice_2 = memory_management.reserve(small_slice_size);
        let _small_slice_3 = memory_management.reserve(small_slice_size);

        assert_eq!(memory_management.chunks.len(), 1);
        assert_eq!(memory_management.slices.len(), 3);
        for (.., slice) in memory_management.slices.iter() {
            assert_eq!(slice.storage.size(), 32);
        }
    }

    #[test]
    fn when_defragmentation_called_if_two_slices_free_should_merge_into_bigger_slice() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::new_period_tick(1),
            SliceStrategy::Ratio(0.2),
        );

        let chunk_handle = memory_management.create_chunk(32 + 32);
        let slice = memory_management.create_slice(0, 32 + 32, chunk_handle.clone());
        memory_management.insert_slice(slice, *chunk_handle.id());

        let _slice_1 = memory_management.reserve(32);
        let _slice_2 = memory_management.reserve(32);

        assert_eq!(memory_management.chunks.len(), 1);
        assert_eq!(memory_management.slices.len(), 2);
        for (.., slice) in memory_management.slices.iter() {
            assert_eq!(slice.storage.size(), 32);
        }
    }

    #[test]
    fn when_defragmentation_called_should_merge_contiguous_free_slice_together() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::new_period_tick(1),
            SliceStrategy::Ratio(0.1),
        );

        //The chunk will be separated in 7 slices. 1 free, 23 not free, 456 free and 7 not free
        let slice_size = 32;
        let num_of_slice = 7;

        let chunk_handle = memory_management.create_chunk(slice_size * num_of_slice);
        let chunk_id = *chunk_handle.id();
        let slice = memory_management.create_slice(0, slice_size * num_of_slice, chunk_handle);
        memory_management.insert_slice(slice, chunk_id);

        let _slice_1 = memory_management.reserve(slice_size);
        let _slice_2 = memory_management.reserve(slice_size);
        let _slice_3 = memory_management.reserve(slice_size);
        let _slice_4 = memory_management.reserve(slice_size);
        let _slice_5 = memory_management.reserve(slice_size);
        let _slice_6 = memory_management.reserve(slice_size);
        let _slice_7 = memory_management.reserve(slice_size);
        drop(_slice_1);
        drop(_slice_4);
        drop(_slice_5);
        drop(_slice_6);
        memory_management.defragmentation();

        assert_eq!(memory_management.chunks.len(), 1);
        assert_eq!(memory_management.slices.len(), 5);

        let chunk = memory_management.chunks.get(&chunk_id).unwrap();
        let slices = &chunk.slices;

        // first slice test
        let first_slice_id = slices.first().unwrap();
        let first_slice = memory_management.slices.get(first_slice_id).unwrap();
        assert!(first_slice.handle.is_free());
        assert_eq!(first_slice.storage.size(), slice_size);
        assert_eq!(first_slice.storage.offset(), 0);

        // second slice test
        let first_slice_id = slices.get(1).unwrap();
        let first_slice = memory_management.slices.get(first_slice_id).unwrap();
        assert!(!first_slice.handle.is_free());
        assert_eq!(first_slice.storage.size(), slice_size);
        assert_eq!(first_slice.storage.offset(), slice_size);

        // third slice test
        let first_slice_id = slices.get(2).unwrap();
        let first_slice = memory_management.slices.get(first_slice_id).unwrap();
        assert!(!first_slice.handle.is_free());
        assert_eq!(first_slice.storage.size(), slice_size);
        assert_eq!(first_slice.storage.offset(), slice_size * 2);

        // fourth slice test (456 merged)
        let first_slice_id = slices.get(3).unwrap();
        let first_slice = memory_management.slices.get(first_slice_id).unwrap();
        assert!(first_slice.handle.is_free());
        assert_eq!(first_slice.storage.size(), slice_size * 3);
        assert_eq!(first_slice.storage.offset(), slice_size * 3);

        // fifth slice test
        let first_slice_id = slices.get(4).unwrap();
        let first_slice = memory_management.slices.get(first_slice_id).unwrap();
        assert!(!first_slice.handle.is_free());
        assert_eq!(first_slice.storage.size(), slice_size);
        assert_eq!(first_slice.storage.offset(), slice_size * 6);
    }

    #[test]
    fn never_dealloc_strategy_never_deallocs() {
        let mut never_dealloc = MergingStrategy::Never;
        for _ in 0..20 {
            assert!(!never_dealloc.should_perform_defragmentation())
        }
    }

    #[test]
    fn period_tick_dealloc_strategy_should_dealloc_after_period() {
        let period = 3;
        let mut period_tick_dealloc = MergingStrategy::new_period_tick(period);

        for _ in 0..3 {
            for _ in 0..period - 1 {
                assert!(!period_tick_dealloc.should_perform_defragmentation());
            }
            assert!(period_tick_dealloc.should_perform_defragmentation());
        }
    }

    #[test]
    fn slice_strategy_minimum_bytes() {
        let strategy = SliceStrategy::MinimumSize(100);

        assert!(strategy.can_use_chunk(200, 101));
        assert!(!strategy.can_use_chunk(200, 99));
    }

    #[test]
    fn slice_strategy_maximum_bytes() {
        let strategy = SliceStrategy::MaximumSize(100);

        assert!(strategy.can_use_chunk(200, 99));
        assert!(!strategy.can_use_chunk(200, 101));
    }

    #[test]
    fn slice_strategy_ratio() {
        let strategy = SliceStrategy::Ratio(0.9);

        assert!(strategy.can_use_chunk(200, 180));
        assert!(!strategy.can_use_chunk(200, 179));
    }

    #[test]
    fn test_handle_mutability() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::Never,
            SliceStrategy::Ratio(0.5),
        );
        let handle = memory_management.reserve(10);

        let other_ref = handle.clone();

        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test]
    fn support_multiple_slices_for_a_chunk() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::Never,
            SliceStrategy::Ratio(0.2),
        );
        let handle = memory_management.reserve(100);
        core::mem::drop(handle);

        let _slice_1 = memory_management.reserve(30);
        let _slice_2 = memory_management.reserve(30);
        let _slice_3 = memory_management.reserve(30);

        assert_eq!(memory_management.chunks.len(), 1);
        assert_eq!(memory_management.slices.len(), 4);
    }

    #[test]
    fn test_slice_mutability() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            MergingStrategy::Never,
            SliceStrategy::Ratio(0.5),
        );
        let first_slice = memory_management.reserve(10);

        drop(first_slice);

        let slice = memory_management.reserve(8);

        if let super::DynamicHandle::Slice(slice) = slice {
            let other_ref = slice.clone();

            assert!(
                !slice.can_mut(),
                "Slice can't be mut when multiple ref to the same handle."
            );
            drop(other_ref);
            assert!(
                slice.can_mut(),
                "Slice should be mut when only one ref to the same handle."
            );
            assert!(
                !slice.is_free(),
                "Slice can't be reallocated when one ref still exist."
            );
        }
    }
}
