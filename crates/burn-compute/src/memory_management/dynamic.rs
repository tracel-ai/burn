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

/// The strategy defines the frequency at which deallocation of unused memory chunks should occur.
#[derive(Debug)]
pub enum DeallocStrategy {
    /// Once every n calls to reserve.
    PeriodTick {
        /// Number of calls to be executed before triggering the deallocation.
        period: usize,
        /// Current state. Should start at zero.
        state: usize,
    },
    #[cfg(feature = "std")]
    /// Once every period of time
    PeriodTime {
        /// Number of time before triggering the deallocation.
        period: time::Duration,
        /// Current state. Should start at now.
        state: time::Instant,
    },
    /// Never deallocate.
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

impl DeallocStrategy {
    /// Create a new strategy with the given period.
    pub fn new_period_tick(period: usize) -> Self {
        DeallocStrategy::PeriodTick { period, state: 0 }
    }

    fn should_dealloc(&mut self) -> bool {
        match self {
            DeallocStrategy::PeriodTick { period, state } => {
                *state = (*state + 1) % *period;
                *state == 0
            }
            #[cfg(feature = "std")]
            DeallocStrategy::PeriodTime { period, state } => {
                if &state.elapsed() > period {
                    *state = time::Instant::now();
                    true
                } else {
                    false
                }
            }
            DeallocStrategy::Never => false,
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
    // It is important to keep the chunk handle inside the slice, since it increases the ref count
    // on the chunk id and make the `is_free` method return false until the slice is freed.
    //
    // TL;DR we can't only store the chunk id.
    chunk: ChunkHandle,
    padding: usize,
}

impl Slice {
    pub fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct DynamicMemoryManagement<Storage> {
    chunks: HashMap<ChunkId, Chunk>,
    slices: HashMap<SliceId, Slice>,
    dealloc_strategy: DeallocStrategy,
    slice_strategy: SliceStrategy,
    storage: Storage,
}

impl<Storage> core::fmt::Debug for DynamicMemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "DynamicMemoryManagement {:?} - {:?}",
                self.dealloc_strategy,
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
    /// Also clean ups, removing unused slices, and chunks if permitted by deallocation strategy.
    fn reserve(&mut self, size: usize) -> Self::Handle {
        let handle = self.reserve_algorithm(size);

        if self.dealloc_strategy.should_dealloc() {
            self.cleanup_chunks();
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
    /// Creates a new instance using the given storage, deallocation strategy and slice strategy.
    pub fn new(
        storage: Storage,
        dealloc_strategy: DeallocStrategy,
        slice_strategy: SliceStrategy,
    ) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            dealloc_strategy,
            slice_strategy,
            storage,
        }
    }

    fn reserve_algorithm(&mut self, size: usize) -> DynamicHandle {
        // Looks for a large enough, existing but unused chunk of memory.
        let slice = self.find_free_slice(size);

        match slice {
            Some(slice) => DynamicHandle::Slice(slice.clone()),
            None => self.alloc(size),
        }
    }

    /// Finds the smallest of the free and large enough chunks to fit `size`
    /// Returns the chunk's id and size.
    fn find_free_slice(&mut self, size: usize) -> Option<SliceHandle> {
        let requires_full_chunk = size < 16;
        let padding = calculate_padding(size);
        let effective_size = size + padding;

        let mut size_diff_current = usize::MAX;

        let mut found = None;

        log::info!("Num chunks {}", self.chunks.len());
        for (chunk_id, chunk) in self.chunks.iter() {
            if requires_full_chunk {
                if self.slices.len() > 1 {
                    continue;
                }
            } else if !self
                .slice_strategy
                .can_use_chunk(chunk.storage.size(), effective_size)
            {
                // continue;
            }

            for (position, slice_id) in chunk.slices.iter().enumerate() {
                if let Some(slice) = self.slices.get(slice_id) {
                    if slice.handle.is_free() {
                        let storage_size = slice.effective_size();
                        let is_big_enough = storage_size >= effective_size;

                        if is_big_enough {
                            let size_diff = storage_size - effective_size;

                            if size_diff < size_diff_current {
                                found = Some((position, *chunk_id, *slice_id));
                                size_diff_current = storage_size - effective_size;
                            }
                        }
                    }
                }
            }
        }

        let (position, chunk_id, slice_id) = match found {
            Some(val) => val,
            None => return None,
        };

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

        assert!(size_diff_current % 32 == 0);

        let chunk = self.chunks.get_mut(&chunk_id).unwrap();
        let handle_chunk = chunk.handle.clone();

        let mut slices = Vec::with_capacity(chunk.slices.len() + 1);
        let mut offset = 0;

        let mut slices_old = Vec::new();
        core::mem::swap(&mut slices_old, &mut chunk.slices);

        let mut handle = None;
        for (pos, slice_id) in slices_old.into_iter().enumerate() {
            // Assumes that all slices are contiguous in a chunk.
            let slice = self.slices.get(&slice_id).unwrap();

            if pos != position {
                slices.push(slice_id);
                offset += slice.effective_size();
                continue;
            }

            let slice_start = self.create_slice(offset, size, handle_chunk.clone());
            let slice_start_id = *slice_start.handle.id();
            offset += slice_start.effective_size();

            let slice_end = self.create_slice(offset, size_diff_current, handle_chunk.clone());
            let slice_end_id = *slice_end.handle.id();
            offset += slice_end.effective_size();

            let created_offset = slice_start.effective_size() + slice_end.effective_size();
            assert_eq!(created_offset, slice.effective_size());

            handle = Some(slice_start.handle.clone());
            self.slices.insert(slice_start_id, slice_start);
            self.slices.insert(slice_end_id, slice_end);

            slices.push(slice_start_id);
            slices.push(slice_end_id);
        }

        let chunk = self.chunks.get_mut(&chunk_id).unwrap();
        self.slices.remove(&slice_id);
        chunk.slices = slices;

        handle
    }

    /// Creates a slice of size `size` upon the given chunk.
    ///
    /// For now slices must start at zero, therefore there can be only one per chunk
    fn create_slice(&self, offset: usize, size: usize, handle_chunk: ChunkHandle) -> Slice {
        let handle_id = *handle_chunk.value.id();
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
    fn create_chunk(&mut self, size: usize) -> ChunkHandle {
        let padding = calculate_padding(size);
        let effective_size = size + padding;

        let storage = self.storage.alloc(effective_size);
        let handle = ChunkHandle::new();

        self.chunks.insert(
            *handle.id(),
            Chunk::new(storage, handle.clone(), Vec::new()),
        );

        handle
    }

    /// Deallocates free chunks and remove them from chunks map.
    fn cleanup_chunks(&mut self) {
        let mut ids_to_remove = Vec::new();

        self.chunks.iter().for_each(|(chunk_id, chunk)| {
            let mut can_dealloc = true;
            for slice in chunk.slices.iter() {
                let slice = self.slices.get(slice).unwrap();

                if !slice.handle.is_free() {
                    can_dealloc = false;
                }
            }

            if can_dealloc {
                ids_to_remove.push(*chunk_id);
            }
        });

        ids_to_remove
            .iter()
            .map(|chunk_id| self.chunks.remove(chunk_id).unwrap())
            .for_each(|chunk| {
                self.storage.dealloc(chunk.storage.id);
            });
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
            DeallocStrategy::Never,
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
            DeallocStrategy::Never,
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
            DeallocStrategy::Never,
            SliceStrategy::Never,
        );
        let chunk_size = 4;
        let _chunk_handle = memory_management.reserve(chunk_size);
        let _new_handle = memory_management.reserve(chunk_size);

        assert_eq!(memory_management.chunks.len(), 2);
    }

    #[test]
    fn when_empty_chunk_is_cleaned_upexists_it_disappears() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            DeallocStrategy::Never,
            SliceStrategy::Never,
        );
        let chunk_size = 4;
        let chunk_handle = memory_management.reserve(chunk_size);
        drop(chunk_handle);
        memory_management.cleanup_chunks();

        assert_eq!(memory_management.chunks.len(), 0);
    }

    #[test]
    fn never_dealloc_strategy_never_deallocs() {
        let mut never_dealloc = DeallocStrategy::Never;
        for _ in 0..20 {
            assert!(!never_dealloc.should_dealloc())
        }
    }

    #[test]
    fn period_tick_dealloc_strategy_should_dealloc_after_period() {
        let period = 3;
        let mut period_tick_dealloc = DeallocStrategy::new_period_tick(period);

        for _ in 0..3 {
            for _ in 0..period - 1 {
                assert!(!period_tick_dealloc.should_dealloc());
            }
            assert!(period_tick_dealloc.should_dealloc());
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
            DeallocStrategy::Never,
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
            DeallocStrategy::Never,
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
            DeallocStrategy::Never,
            SliceStrategy::Ratio(0.5),
        );
        let chunk = memory_management.reserve(10);

        if let super::DynamicHandle::Slice(_) = chunk {
            panic!("Should be a chunk.")
        }

        drop(chunk);

        let slice = memory_management.reserve(8);

        if let super::DynamicHandle::Chunk(_) = &slice {
            panic!("Should be a slice.")
        }

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

fn calculate_padding(size: usize) -> usize {
    let rem = size % 32;
    if rem != 0 {
        32 - rem
    } else {
        0
    }
}
