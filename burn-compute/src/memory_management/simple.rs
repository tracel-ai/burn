use crate::{id_type, ComputeStorage, MemoryHandle, MemoryManagement, MemorySpace, StorageHandle};
use alloc::sync::Arc;
use std::collections::HashMap;

id_type!(ChunkId);
id_type!(SliceId);

#[derive(Clone)]
pub enum SimpleHandle {
    Chunk(ChunkId),
    Slice(SliceId),
}

pub struct SimpleMemoryManagement<Storage> {
    chunks: HashMap<ChunkId, (StorageHandle, Vec<SliceId>)>,
    slices: HashMap<SliceId, (StorageHandle, ChunkId)>,
    storage: Storage,
    frequency: usize,
    current: usize,
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

    fn get(&mut self, handle: &Self::Handle) -> Storage::StorageResource {
        let resource = match handle {
            SimpleHandle::Chunk(id) => &self.chunks.get(id).unwrap().0,
            SimpleHandle::Slice(id) => &self.slices.get(id).unwrap().0,
        };

        self.storage.get(resource)
    }

    fn reserve(&mut self, size: usize) -> Self::Handle {
        let mut size_diff_current = usize::MAX;
        let mut current = None;

        self.chunks
            .iter_mut()
            .for_each(|(key, (ressource, slices))| {
                let is_free = slices.is_empty() && Arc::strong_count(&key.id) == 1;
                let space_available = match ressource.space {
                    MemorySpace::Full(len) => len,
                    MemorySpace::Slice(_, _) => panic!("Only chunks"),
                };

                let size_diff = space_available - size;
                if is_free && size_diff < size_diff_current {
                    current = Some((key, ressource, slices));
                    size_diff_current = size_diff;
                }
            });

        let ressource = match current {
            Some((id, ressource, slices)) => {
                if size_diff_current == 0 {
                    return SimpleHandle::Chunk(id.clone());
                }
                let slice_id = SliceId::new();
                let ressource = StorageHandle {
                    id: ressource.id.clone(),
                    space: MemorySpace::Slice(0, size),
                };

                slices.push(slice_id.clone());
                self.slices
                    .insert(slice_id.clone(), (ressource, id.clone()));

                return SimpleHandle::Slice(slice_id);
            }
            None => self.create_new(size),
        };

        self.cleanup();
        ressource
    }
}

impl<Storage: ComputeStorage> SimpleMemoryManagement<Storage> {
    pub fn new(storage: Storage) -> Self {
        Self {
            storage,
            chunks: HashMap::new(),
            slices: HashMap::new(),
            frequency: 100,
            current: 0,
        }
    }
    fn create_new(&mut self, size: usize) -> SimpleHandle {
        let ressource = self.storage.alloc(size);
        let chunk_id = ChunkId::new();

        self.chunks
            .insert(chunk_id.clone(), (ressource, Vec::new()));

        SimpleHandle::Chunk(chunk_id)
    }

    fn cleanup(&mut self) {
        self.current += 1;

        if self.current > self.frequency {
            self.cleanup_slices();
            self.cleanup_chunks();
            self.current = 0;
        }
    }

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
