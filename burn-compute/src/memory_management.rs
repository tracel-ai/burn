use crate::{ComputeStorage, MemorySpace, ResourceDescription};
use alloc::sync::Arc;
use std::collections::HashMap;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            id: Arc<String>,
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    id: Arc::new(burn_common::id::IdGenerator::generate()),
                }
            }
        }
    };
}

id_type!(ChunkId);
id_type!(SliceId);

pub enum Resource {
    Chunk(ChunkId),
    Slice(SliceId),
}

impl Resource {
    pub fn can_mut(&self) -> bool {
        match self {
            Resource::Chunk(id) => Arc::strong_count(&id.id) == 3,
            Resource::Slice(id) => Arc::strong_count(&id.id) == 3,
        }
    }
}

pub struct BasicMemoryManagement<Storage> {
    chunks: HashMap<ChunkId, (ResourceDescription, Vec<SliceId>)>,
    slices: HashMap<SliceId, (ResourceDescription, ChunkId)>,
    storage: Storage,
    frequency: usize,
    current: usize,
}

pub trait MemoryManagement<Storage: ComputeStorage> {
    type Resource;

    fn get(&mut self, ressource: &Self::Resource) -> Storage::StorageResource;
    fn reserve(&mut self, size: usize) -> Self::Resource;
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> for BasicMemoryManagement<Storage> {
    type Resource = Resource;

    fn get(&mut self, ressource: &Self::Resource) -> Storage::StorageResource {
        let ressource = match ressource {
            Resource::Chunk(id) => &self.chunks.get(id).unwrap().0,
            Resource::Slice(id) => &self.slices.get(id).unwrap().0,
        };

        let ressource = self.storage.get(ressource);
        ressource
    }

    fn reserve(&mut self, size: usize) -> Self::Resource {
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
                    return Resource::Chunk(id.clone());
                }
                let slice_id = SliceId::new();
                let ressource = ResourceDescription {
                    id: ressource.id.clone(),
                    space: MemorySpace::Slice(0, size),
                };

                slices.push(slice_id.clone());
                self.slices
                    .insert(slice_id.clone(), (ressource, id.clone()));

                return Resource::Slice(slice_id);
            }
            None => self.create_new(size),
        };

        self.cleanup();
        ressource
    }
}

impl<Storage: ComputeStorage> BasicMemoryManagement<Storage> {
    pub fn new(storage: Storage) -> Self {
        Self {
            storage,
            chunks: HashMap::new(),
            slices: HashMap::new(),
            frequency: 100,
            current: 0,
        }
    }
    fn create_new(&mut self, size: usize) -> Resource {
        let ressource = self.storage.alloc(size);
        let chunk_id = ChunkId::new();

        self.chunks
            .insert(chunk_id.clone(), (ressource, Vec::new()));

        Resource::Chunk(chunk_id)
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
