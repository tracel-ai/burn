use crate::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use alloc::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;

#[derive(Default)]
pub struct BytesStorage {
    memory: HashMap<StorageId, AllocatedBytes>,
}

pub struct BytesResource {
    ptr: *mut u8,
    utilization: StorageUtilization,
}

struct AllocatedBytes {
    ptr: *mut u8,
    layout: Layout,
}

impl BytesResource {
    pub fn write<'a>(&self) -> &'a mut [u8] {
        let (ptr, len) = match self.utilization {
            StorageUtilization::Full(len) => (self.ptr, len),
            StorageUtilization::Slice(location, len) => unsafe { (self.ptr.add(location), len) },
        };

        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    pub fn read<'a>(&self) -> &'a [u8] {
        let (ptr, len) = match self.utilization {
            StorageUtilization::Full(len) => (self.ptr, len),
            StorageUtilization::Slice(location, len) => unsafe { (self.ptr.add(location), len) },
        };

        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

impl ComputeStorage for BytesStorage {
    type Resource = BytesResource;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let memory = self.memory.get_mut(&handle.id).unwrap();

        BytesResource {
            ptr: memory.ptr,
            utilization: handle.utilization.clone(),
        }
    }

    fn alloc(&mut self, size: usize) -> StorageHandle {
        let id = StorageId::new();
        let handle = StorageHandle {
            id: id.clone(),
            utilization: StorageUtilization::Full(size),
        };

        unsafe {
            let layout = Layout::array::<u8>(size).unwrap();
            let ptr = alloc(layout.clone());
            let memory = AllocatedBytes { ptr, layout };

            self.memory.insert(id, memory);
        }

        handle
    }

    fn dealloc(&mut self, handle: &StorageHandle) {
        if let Some(memory) = self.memory.remove(&handle.id) {
            unsafe {
                dealloc(memory.ptr, memory.layout);
            }
        }
    }
}
