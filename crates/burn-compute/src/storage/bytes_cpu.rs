use super::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use alloc::alloc::{alloc, dealloc, Layout};
use hashbrown::HashMap;

/// The bytes storage maps ids to pointers of bytes in a contiguous layout.
#[derive(Default)]
pub struct BytesStorage {
    memory: HashMap<StorageId, AllocatedBytes>,
}

impl core::fmt::Debug for BytesStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("BytesStorage")
    }
}

/// Can send to other threads.
unsafe impl Send for BytesStorage {}
unsafe impl Send for BytesResource {}

/// This struct is a pointer to a memory chunk or slice.
pub struct BytesResource {
    ptr: *mut u8,
    utilization: StorageUtilization,
}

/// This struct refers to a specific (contiguous) layout of bytes.
struct AllocatedBytes {
    ptr: *mut u8,
    layout: Layout,
}

impl BytesResource {
    fn get_exact_location_and_length(&self) -> (*mut u8, usize) {
        match self.utilization {
            StorageUtilization::Full(len) => (self.ptr, len),
            StorageUtilization::Slice { offset, size } => unsafe { (self.ptr.add(offset), size) },
        }
    }

    /// Returns the resource as a mutable slice of bytes.
    pub fn write<'a>(&self) -> &'a mut [u8] {
        let (ptr, len) = self.get_exact_location_and_length();

        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Returns the resource as an immutable slice of bytes.
    pub fn read<'a>(&self) -> &'a [u8] {
        let (ptr, len) = self.get_exact_location_and_length();

        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

impl ComputeStorage for BytesStorage {
    type Resource = BytesResource;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let allocated_bytes = self.memory.get_mut(&handle.id).unwrap();

        BytesResource {
            ptr: allocated_bytes.ptr,
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
            let ptr = alloc(layout);
            let memory = AllocatedBytes { ptr, layout };

            self.memory.insert(id, memory);
        }

        handle
    }

    fn dealloc(&mut self, id: StorageId) {
        if let Some(memory) = self.memory.remove(&id) {
            unsafe {
                dealloc(memory.ptr, memory.layout);
            }
        }
    }

    fn copy(&mut self, from: &StorageHandle, to: &StorageHandle) {
        assert_eq!(from.size(), to.size());

        let input = self.get(from);
        let output = self.get(to);

        for i in 0..from.size() {
            let offset = i + from.offset();
            let ptr_out = output.ptr.wrapping_add(offset);

            let offset = i + to.offset();
            let ptr_in = input.ptr.wrapping_add(offset);

            unsafe { *ptr_in = *ptr_out }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_alloc_and_dealloc() {
        let mut storage = BytesStorage::default();
        let handle_1 = storage.alloc(64);

        assert_eq!(handle_1.size(), 64);
        storage.dealloc(handle_1.id);
    }

    #[test]
    fn test_slices() {
        let mut storage = BytesStorage::default();
        let handle_1 = storage.alloc(64);
        let handle_2 = StorageHandle::new(
            handle_1.id.clone(),
            StorageUtilization::Slice {
                offset: 24,
                size: 8,
            },
        );

        storage
            .get(&handle_1)
            .write()
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| {
                *b = i as u8;
            });

        let bytes = storage.get(&handle_2).read().to_vec();
        storage.dealloc(handle_1.id);
        assert_eq!(bytes, &[24, 25, 26, 27, 28, 29, 30, 31]);
    }
}
