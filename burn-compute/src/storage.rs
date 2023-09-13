use crate::id_type;
use std::collections::HashMap;

id_type!(StorageId);

#[derive(Clone)]
pub enum MemorySpace {
    Full(usize),
    Slice(usize, usize),
}

#[derive(Clone)]
pub struct StorageHandle {
    pub id: StorageId,
    pub space: MemorySpace,
}

pub trait ComputeStorage {
    type StorageResource;

    fn get(&mut self, handle: &StorageHandle) -> Self::StorageResource;
    fn alloc(&mut self, size: usize) -> StorageHandle;
    fn dealloc(&mut self, handle: &StorageHandle);
}

//// IMPL

#[derive(Default)]
pub struct BytesStorage {
    data: HashMap<StorageId, Vec<u8>>,
}

pub struct BytesResource {
    ptr: *mut u8,
    space: MemorySpace,
}

impl BytesResource {
    pub fn write<'a>(&self) -> &'a mut [u8] {
        let (ptr, len) = match self.space {
            MemorySpace::Full(len) => (self.ptr, len),
            MemorySpace::Slice(location, len) => unsafe { (self.ptr.add(location), len) },
        };

        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    pub fn read<'a>(&self) -> &'a [u8] {
        let (ptr, len) = match self.space {
            MemorySpace::Full(len) => (self.ptr, len),
            MemorySpace::Slice(location, len) => unsafe { (self.ptr.add(location), len) },
        };

        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

impl ComputeStorage for BytesStorage {
    type StorageResource = BytesResource;

    fn get(&mut self, description: &StorageHandle) -> Self::StorageResource {
        let ptr = self.data.get_mut(&description.id).unwrap().as_mut_ptr();

        BytesResource {
            ptr,
            space: description.space.clone(),
        }
    }

    fn alloc(&mut self, size: usize) -> StorageHandle {
        let id = StorageId::new();
        let ressource = StorageHandle {
            id: id.clone(),
            space: MemorySpace::Full(size),
        };
        self.data.insert(id, vec![0; size]);

        ressource
    }

    fn dealloc(&mut self, description: &StorageHandle) {
        self.data.remove(&description.id);
    }
}
