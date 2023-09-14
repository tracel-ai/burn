use super::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use alloc::alloc::{alloc, dealloc, Layout};
use hashbrown::HashMap;

/// The BytesStorage maps ids to pointers of bytes in a contiguous layout
#[derive(Default)]
pub struct BytesStorage {
    memory: HashMap<StorageId, AllocatedBytes>,
}

/// Can send to other thread, but can't sync.
unsafe impl Send for BytesStorage {}
unsafe impl Send for BytesResource {}

/// The BytesResource struct is a pointer to a memory chunk or slice
pub struct BytesResource {
    ptr: *mut u8,
    utilization: StorageUtilization,
}

/// The AllocatedBytes struct refers to a specific (contiguous) layout of bytes
struct AllocatedBytes {
    ptr: *mut u8,
    layout: Layout,
}

impl BytesResource {
    fn get_exact_location_and_length(&self) -> (*mut u8, usize) {
        match self.utilization {
            StorageUtilization::Full(len) => (self.ptr, len),
            StorageUtilization::Slice(location, len) => unsafe { (self.ptr.add(location), len) },
        }
    }

    /// Returns the resource as a mutable slice of bytes
    pub fn write<'a>(&self) -> &'a mut [u8] {
        let (ptr, len) = self.get_exact_location_and_length();

        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Returns the resource as an immutable slice of bytes
    pub fn read<'a>(&self) -> &'a [u8] {
        let (ptr, len) = self.get_exact_location_and_length();

        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

impl ComputeStorage for BytesStorage {
    type Resource = BytesResource;

    /// Returns the bytes corresponding to a handle
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let allocated_bytes = self.memory.get_mut(&handle.id).unwrap();

        BytesResource {
            ptr: allocated_bytes.ptr,
            utilization: handle.utilization.clone(),
        }
    }

    /// Allocates `size` bytes of memory and creates a handle refering to them
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

    /// Deallocates the memory referred by the handle
    fn dealloc(&mut self, handle: &StorageHandle) {
        if let Some(memory) = self.memory.remove(&handle.id) {
            unsafe {
                dealloc(memory.ptr, memory.layout);
            }
        }
    }
}
