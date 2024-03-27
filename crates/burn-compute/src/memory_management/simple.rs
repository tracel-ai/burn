use super::{MemoryHandle, MemoryManagement};
use crate::{
    memory_id_type,
    storage::{ComputeStorage, StorageHandle, StorageUtilization},
};
use alloc::{sync::Arc, vec::Vec};
use hashbrown::HashMap;

#[cfg(all(not(target_family = "wasm"), feature = "std"))]
use std::time;
#[cfg(all(target_family = "wasm", feature = "std"))]
use web_time as time;

// The ChunkId allows to keep track of how many references there are to a specific chunk.
memory_id_type!(ChunkId);
// The SliceId allows to keep track of how many references there are to a specific slice.
memory_id_type!(SliceId);

impl ChunkId {
    /// A chunk is free if it is only referred by the chunk hashmap.
    fn is_free(&self) -> bool {
        Arc::strong_count(&self.id) <= 1
    }
}

impl SliceId {
    /// A slice is free if it is only referred by the slice hashmap and the chunk it is in.
    fn is_free(&self) -> bool {
        Arc::strong_count(&self.id) <= 2
    }
}

/// The SimpleHandle is a memory handle, referring to either a chunk or a slice.
#[derive(Debug, Clone)]
pub enum SimpleHandle {
    /// A whole chunk of memory.
    Chunk(ChunkId),
    /// A slice of a chunk of memory.
    Slice(SliceId),
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

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct SimpleMemoryManagement<Storage> {
    chunks: HashMap<ChunkId, (StorageHandle, Vec<SliceId>)>,
    slices: HashMap<SliceId, (StorageHandle, ChunkId)>,
    dealloc_strategy: DeallocStrategy,
    slice_strategy: SliceStrategy,
    storage: Storage,
}

impl<Storage> core::fmt::Debug for SimpleMemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "SimpleMemoryManagement {:?} - {:?}",
                self.dealloc_strategy,
                core::any::type_name::<Storage>(),
            )
            .as_str(),
        )
    }
}

impl MemoryHandle for SimpleHandle {
    /// Returns true if referenced by only one tensor, and only once by the
    /// memory management hashmaps
    fn can_mut(&self) -> bool {
        // One reference in the chunk hashmap, another owned by one tensor.
        const REFERENCE_LIMIT_CHUNK: usize = 2;
        // One reference in the chunk hashmap (for the chunk on which this slice is built),
        // another in the slice hashmap for this slice, and another owned by one tensor.
        const REFERENCE_LIMIT_SLICE: usize = 3;

        match &self {
            SimpleHandle::Chunk(id) => Arc::strong_count(&id.id) <= REFERENCE_LIMIT_CHUNK,
            SimpleHandle::Slice(id) => Arc::strong_count(&id.id) <= REFERENCE_LIMIT_SLICE,
        }
    }
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> for SimpleMemoryManagement<Storage> {
    type Handle = SimpleHandle;

    /// Returns the resource from the storage, for the specified handle.
    fn get(&mut self, handle: &Self::Handle) -> Storage::Resource {
        let resource = match &handle {
            SimpleHandle::Chunk(id) => &self.chunks.get(id).unwrap().0,
            SimpleHandle::Slice(id) => &self.slices.get(id).unwrap().0,
        };

        self.storage.get(resource)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, removing unused slices, and chunks if permitted by deallocation strategy.
    fn reserve(&mut self, size: usize) -> Self::Handle {
        self.cleanup_slices();

        let handle = self.reserve_algorithm(size);

        if self.dealloc_strategy.should_dealloc() {
            self.cleanup_chunks();
        }

        handle
    }

    fn alloc(&mut self, size: usize) -> Self::Handle {
        self.create_chunk(size)
    }

    fn dealloc(&mut self, handle: &Self::Handle) {
        match handle {
            SimpleHandle::Chunk(id) => {
                if let Some((handle, _slices)) = self.chunks.remove(id) {
                    self.storage.dealloc(handle.id);
                }
            }
            SimpleHandle::Slice(_) => panic!("Can't dealloc slice manually"),
        }
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }
}

impl<Storage: ComputeStorage> SimpleMemoryManagement<Storage> {
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

    fn reserve_algorithm(&mut self, size: usize) -> SimpleHandle {
        // Looks for a large enough, existing but unused chunk of memory.
        let chunk = self.find_free_chunk(size);

        match chunk {
            Some((chunk_id, chunk_size)) => {
                if size == chunk_size {
                    // If there is one of exactly the same size, it reuses it.
                    SimpleHandle::Chunk(chunk_id.clone())
                } else {
                    // Otherwise creates a slice of the right size upon it, always starting at zero.
                    self.create_slice(size, chunk_id)
                }
            }
            // If no chunk available, creates one of exactly the right size.
            None => self.create_chunk(size),
        }
    }

    /// Finds the smallest of the free and large enough chunks to fit `size`
    /// Returns the chunk's id and size.
    fn find_free_chunk(&self, size: usize) -> Option<(ChunkId, usize)> {
        let mut size_diff_current = usize::MAX;
        let mut current = None;

        for (chunk_id, (resource, slices)) in self.chunks.iter() {
            // If chunk is already used, we do not choose it
            if !slices.is_empty() || !chunk_id.is_free() {
                continue;
            }

            let resource_size = resource.size();

            // If we find a chunk of exactly the right size, we stop searching altogether
            if size == resource_size {
                current = Some((chunk_id, resource));
                break;
            }

            // Finds the smallest of the large enough chunks that can accept a slice
            // of the given size
            if self.slice_strategy.can_use_chunk(resource_size, size) {
                let size_diff = resource_size - size;

                if size_diff < size_diff_current {
                    current = Some((chunk_id, resource));
                    size_diff_current = size_diff;
                }
            }
        }

        current.map(|(id, handle)| (id.clone(), handle.size()))
    }

    /// Creates a slice of size `size` upon the given chunk.
    ///
    /// For now slices must start at zero, therefore there can be only one per chunk
    fn create_slice(&mut self, size: usize, chunk_id: ChunkId) -> SimpleHandle {
        let (handle, slices) = self.chunks.get_mut(&chunk_id).unwrap();
        let slice_id = SliceId::new();

        let storage = StorageHandle {
            id: handle.id.clone(),
            utilization: StorageUtilization::Slice(0, size),
        };

        if slices.is_empty() {
            self.slices.insert(slice_id.clone(), (storage, chunk_id));
        } else {
            panic!("Can't have more than 1 slice yet.");
        }

        slices.push(slice_id.clone());

        SimpleHandle::Slice(slice_id)
    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk(&mut self, size: usize) -> SimpleHandle {
        let resource = self.storage.alloc(size);
        let chunk_id = ChunkId::new();

        self.chunks.insert(chunk_id.clone(), (resource, Vec::new()));

        SimpleHandle::Chunk(chunk_id)
    }

    /// Deallocates free chunks and remove them from chunks map.
    fn cleanup_chunks(&mut self) {
        let mut ids_to_remove = Vec::new();

        self.chunks.iter().for_each(|(chunk_id, _resource)| {
            if chunk_id.is_free() {
                ids_to_remove.push(chunk_id.clone());
            }
        });

        ids_to_remove
            .iter()
            .map(|chunk_id| self.chunks.remove(chunk_id).unwrap())
            .for_each(|(resource, _slices)| {
                self.storage.dealloc(resource.id);
            });
    }

    /// Removes free slices from slice map and corresponding chunks.
    fn cleanup_slices(&mut self) {
        let mut ids_to_remove = Vec::new();

        self.slices.iter().for_each(|(slice_id, _resource)| {
            if slice_id.is_free() {
                ids_to_remove.push(slice_id.clone());
            }
        });

        ids_to_remove
            .iter()
            .map(|slice_id| {
                let value = self.slices.remove(slice_id).unwrap();
                (slice_id, value.1)
            })
            .for_each(|(slice_id, chunk_id)| {
                let (_chunk, slices) = self.chunks.get_mut(&chunk_id).unwrap();
                slices.retain(|id| id != slice_id);
            });
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        memory_management::{MemoryHandle, MemoryManagement, SliceStrategy},
        storage::BytesStorage,
    };

    use super::{DeallocStrategy, SimpleMemoryManagement};

    #[test]
    fn can_mut_with_single_tensor_reference() {
        let mut memory_management = SimpleMemoryManagement::new(
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
        let mut memory_management = SimpleMemoryManagement::new(
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
        let mut memory_management = SimpleMemoryManagement::new(
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
        let mut memory_management = SimpleMemoryManagement::new(
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
}
