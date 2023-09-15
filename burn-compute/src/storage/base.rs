use crate::id_type;

// The StorageId can be used to map a handle to its actual data.
id_type!(StorageId);

/// The StorageUtilization defines if data uses a full memory chunk or a slice of it.
#[derive(Clone)]
pub enum StorageUtilization {
    /// Full memory chunk of specified size
    Full(usize),
    /// Slice of memory chunk with start index and size.
    Slice(usize, usize),
}

/// The StorageHandle contains the storage id of a resource and the way it is stored
#[derive(Clone, new)]
pub struct StorageHandle {
    /// Storage id.
    pub id: StorageId,
    /// How the storage is used.
    pub utilization: StorageUtilization,
}

impl StorageHandle {
    /// Returns the size of the resource of the handle
    pub fn size(&self) -> usize {
        match self.utilization {
            StorageUtilization::Full(size) => size,
            StorageUtilization::Slice(_, size) => size,
        }
    }
}

/// The ComputeStorage trait is responsible for allocating and deallocating memory.
pub trait ComputeStorage: Send {
    /// The Resource associated type determines the way data is implemented and how
    /// it can be accessed by kernels
    type Resource: Send;

    /// Returns the underlying resource for a specified storage handle
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;

    /// Allocates `size` units of memory and returns a handle to it
    fn alloc(&mut self, size: usize) -> StorageHandle;

    /// Deallocates the memory pointed by the given storage id.
    fn dealloc(&mut self, id: StorageId);
}
