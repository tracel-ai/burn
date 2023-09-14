use crate::{
    id_type, ComputeStorage, MemoryHandle, MemoryManagement, StorageHandle, StorageUtilization,
};
use alloc::sync::Arc;
use std::collections::HashMap;

id_type!(ChunkId);
id_type!(SliceId);

#[derive(Clone)]
pub enum SimpleHandle {
    Chunk(ChunkId),
    Slice(SliceId),
}

pub enum DeallocStrategy {
    PeriodTick(usize, usize),
    PeriodTime(std::time::Duration, std::time::Instant),
    Never,
}

impl DeallocStrategy {
    fn should_dealloc(&mut self) -> bool {
        match self {
            DeallocStrategy::PeriodTick(period, last) => {
                *last += 1;
                if last > period {
                    *last = 0;
                    true
                } else {
                    false
                }
            }
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

pub struct SimpleMemoryManagement<Storage> {
    chunks: HashMap<ChunkId, (StorageHandle, Vec<SliceId>)>,
    slices: HashMap<SliceId, (StorageHandle, ChunkId)>,
    strategy: DeallocStrategy,
    storage: Storage,
}

impl MemoryHandle for SimpleHandle {
    fn can_mut(&self) -> bool {
        match self {
            // One reference in the chunk hashmap, another owned by the tensor.
            SimpleHandle::Chunk(id) => Arc::strong_count(&id.id) <= 2,
            // One reference in the chunk hashmap, another in the slice hashmap, and another owned by the tensor.
            SimpleHandle::Slice(id) => Arc::strong_count(&id.id) <= 3,
        }
    }
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> for SimpleMemoryManagement<Storage> {
    type Handle = SimpleHandle;

    fn get(&mut self, handle: &Self::Handle) -> Storage::Resource {
        let resource = match handle {
            SimpleHandle::Chunk(id) => &self.chunks.get(id).unwrap().0,
            SimpleHandle::Slice(id) => &self.slices.get(id).unwrap().0,
        };

        self.storage.get(resource)
    }

    fn reserve(&mut self, size: usize) -> Self::Handle {
        self.cleanup_slices();

        let chunk = self.find_free_chunk(size);

        let handle = match chunk {
            Some((chunk_id, chunk_size)) => {
                if size == chunk_size {
                    SimpleHandle::Chunk(chunk_id.clone())
                } else {
                    self.create_slice(size, chunk_id)
                }
            }
            None => self.create_new(size),
        };

        if self.strategy.should_dealloc() {
            self.dealloc_chunks();
        }

        handle
    }
}

impl<Storage: ComputeStorage> SimpleMemoryManagement<Storage> {
    pub fn new(storage: Storage, strategy: DeallocStrategy) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            strategy,
            storage,
        }
    }

    pub fn never_dealloc(storage: Storage) -> Self {
        Self::new(storage, DeallocStrategy::Never)
    }

    fn find_free_chunk(&self, size: usize) -> Option<(ChunkId, usize)> {
        let mut size_diff_current = usize::MAX;
        let mut current = None;

        self.chunks.iter().for_each(|(key, (ressource, slices))| {
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

        SimpleHandle::Slice(slide_id)
    }

    fn create_new(&mut self, size: usize) -> SimpleHandle {
        let ressource = self.storage.alloc(size);
        let chunk_id = ChunkId::new();

        self.chunks
            .insert(chunk_id.clone(), (ressource, Vec::new()));

        SimpleHandle::Chunk(chunk_id)
    }

    fn dealloc_chunks(&mut self) {
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
