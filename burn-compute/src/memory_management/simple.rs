use super::{MemoryHandle, MemoryManagement};
use crate::{
    id_type,
    storage::{ComputeStorage, StorageHandle, StorageUtilization},
};
use alloc::{sync::Arc, vec::Vec};
use core::sync::atomic::AtomicUsize;
use hashbrown::HashMap;

// The ChunkId allows to keep track of how many references there are to a specific chunk
id_type!(ChunkId);
// The SliceId allows to keep track of how many references there are to a specific slice
id_type!(SliceId);

/// The SimpleHandle is a memory handle, referring to either a chunk or a slice
pub struct SimpleHandle {
    id: SimpleHandleId,
    compute_reference: Arc<AtomicUsize>,
}

impl SimpleHandle {
    pub fn new(id: SimpleHandleId) -> SimpleHandle {
        Self {
            id,
            compute_reference: Arc::new(0.into()),
        }
    }
}

#[derive(Clone)]
pub enum SimpleHandleId {
    Chunk(ChunkId),
    Slice(SliceId),
}

/// The DeallocStrategy defines the frequency at which deallocation
/// of unused memory chunks should occur
pub enum DeallocStrategy {
    /// Once every n calls to reserve
    /// First associated data is n, second is the state and should start at 0
    PeriodTick(usize, usize),
    #[cfg(feature = "std")]
    /// Once every period of time
    PeriodTime(std::time::Duration, std::time::Instant),
    /// Never deallocate
    Never,
}

impl DeallocStrategy {
    fn should_dealloc(&mut self) -> bool {
        match self {
            DeallocStrategy::PeriodTick(period, last) => {
                *last = (*last + 1) % *period;
                *last == 0
            }
            #[cfg(feature = "std")]
            DeallocStrategy::PeriodTime(period, last) => {
                if &last.elapsed() > period {
                    *last = std::time::Instant::now();
                    true
                } else {
                    false
                }
            }
            DeallocStrategy::Never => false,
        }
    }
}

/// The SimpleMemoryManagement reserves and keeps track of chunks of memory in the storage,
/// and slices upon these chunks
pub struct SimpleMemoryManagement<Storage> {
    chunks: HashMap<ChunkId, (StorageHandle, Vec<SliceId>)>,
    slices: HashMap<SliceId, (StorageHandle, ChunkId)>,
    dealloc_strategy: DeallocStrategy,
    storage: Storage,
}

impl MemoryHandle for SimpleHandle {
    /// Returns true if referenced by only one tensor, and only once by the
    /// memory management hashmaps
    fn can_mut(&self) -> bool {
        let compute_reference = self
            .compute_reference
            .load(core::sync::atomic::Ordering::Relaxed);

        match &self.id {
            // One reference in the chunk hashmap, another owned by the tensor.
            SimpleHandleId::Chunk(id) => Arc::strong_count(&id.id) <= 2 + compute_reference,
            // One reference in the chunk hashmap, another in the slice hashmap, and another owned by the tensor.
            SimpleHandleId::Slice(id) => Arc::strong_count(&id.id) <= 3 + compute_reference,
        }
    }
    fn tensor_reference(&self) -> Self {
        Self {
            id: self.id.clone(),
            compute_reference: self.compute_reference.clone(),
        }
    }
    fn compute_reference(&self) -> Self {
        self.compute_reference
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        Self {
            id: self.id.clone(),
            compute_reference: self.compute_reference.clone(),
        }
    }
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> for SimpleMemoryManagement<Storage> {
    type Handle = SimpleHandle;

    /// Returns the resource from the storage, for the specified handle
    fn get(&mut self, handle: &Self::Handle) -> Storage::Resource {
        let resource = match &handle.id {
            SimpleHandleId::Chunk(id) => &self.chunks.get(id).unwrap().0,
            SimpleHandleId::Slice(id) => &self.slices.get(id).unwrap().0,
        };

        self.storage.get(resource)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    /// Also clean ups, removing unused slices, and chunks if permitted by deallocation strategy
    fn reserve(&mut self, size: usize) -> Self::Handle {
        self.cleanup_slices();

        let handle = self.reserve_algorithm(size);

        if self.dealloc_strategy.should_dealloc() {
            self.cleanup_chunks();
        }

        handle
    }
}

impl<Storage: ComputeStorage> SimpleMemoryManagement<Storage> {
    /// Creates an empty SimpleMemoryManagement
    pub fn new(storage: Storage, dealloc_strategy: DeallocStrategy) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            dealloc_strategy,
            storage,
        }
    }

    /// Creates an empty SimpleMemoryManagement with no deallocation
    pub fn never_dealloc(storage: Storage) -> Self {
        Self::new(storage, DeallocStrategy::Never)
    }

    /// Looks for a large enough, existing but unused chunk of memory.
    /// If there is none, it creates one of exactly the right size.
    /// If there is one of exactly the same size, it reuses it.
    /// If there are only larger chunks, it takes the smallest of them
    /// and creates a slice of the right size upon it, always starting at zero.
    fn reserve_algorithm(&mut self, size: usize) -> SimpleHandle {
        let chunk = self.find_free_chunk(size);

        let handle = match chunk {
            Some((chunk_id, chunk_size)) => {
                if size == chunk_size {
                    SimpleHandle::new(SimpleHandleId::Chunk(chunk_id.clone()))
                } else {
                    self.create_slice(size, chunk_id)
                }
            }
            None => self.create_chunk(size),
        };

        handle
    }
    /// Finds the smallest of the free and large enough chunks to fit `size`
    /// Returns the chunk's id and size
    fn find_free_chunk(&self, size: usize) -> Option<(ChunkId, usize)> {
        let mut size_diff_current = usize::MAX;
        let mut current = None;

        self.chunks.iter().for_each(|(key, (ressource, slices))| {
            // A chunk is free if no slice is built upon it and no tensor
            // depends on it, i.e. only the memory management map refers to it
            let is_free = slices.is_empty() && Arc::strong_count(&key.id) == 1;

            if is_free && ressource.size() > size {
                let size_diff = ressource.size() - size;
                if size_diff < size_diff_current {
                    current = Some((key, ressource));
                    size_diff_current = size_diff;
                }
            }
        });

        current.map(|(id, handle)| (id.clone(), handle.size()))
    }

    /// Creates a slice of size `size` upon the given chunk
    /// For now slices must start at zero, therefore there can be only one per chunk
    fn create_slice(&mut self, size: usize, chunk_id: ChunkId) -> SimpleHandle {
        let (handle, slices) = self.chunks.get_mut(&chunk_id).unwrap();
        let slide_id = SliceId::new();

        let storage = StorageHandle {
            id: handle.id.clone(),
            utilization: StorageUtilization::Slice(0, size),
        };

        if slices.is_empty() {
            self.slices.insert(slide_id.clone(), (storage, chunk_id));
        } else {
            panic!("Can't have more than 1 slice yet.");
        }

        slices.push(slide_id.clone());

        SimpleHandle::new(SimpleHandleId::Slice(slide_id))
    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk(&mut self, size: usize) -> SimpleHandle {
        let ressource = self.storage.alloc(size);
        let chunk_id = ChunkId::new();

        self.chunks
            .insert(chunk_id.clone(), (ressource, Vec::new()));

        SimpleHandle::new(SimpleHandleId::Chunk(chunk_id))
    }

    /// Deallocates free chunks and remove them from chunks map
    fn cleanup_chunks(&mut self) {
        let mut keys_to_remove = Vec::new();

        self.chunks.iter().for_each(|(key, _ressource)| {
            if Arc::strong_count(&key.id) == 1 {
                keys_to_remove.push(key.clone());
            }
        });

        keys_to_remove
            .into_iter()
            .map(|key| self.chunks.remove(&key).unwrap())
            .for_each(|(ressource, _slices)| {
                self.storage.dealloc(&ressource);
            });
    }

    /// Removes free slices from slice map and corresponding chunks
    fn cleanup_slices(&mut self) {
        let mut keys_to_remove = Vec::new();

        self.slices.iter().for_each(|(key, _ressource)| {
            if Arc::strong_count(&key.id) == 1 {
                keys_to_remove.push(key.clone());
            }
        });

        keys_to_remove
            .into_iter()
            .map(|key| {
                let value = self.slices.remove(&key).unwrap();
                (key, value.1)
            })
            .for_each(|(slice_id, chunk_id)| {
                let (_chunk, slices) = self.chunks.get_mut(&chunk_id).unwrap();
                slices.retain(|id| *id != slice_id);
            });
    }
}
