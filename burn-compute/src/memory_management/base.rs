use crate::storage::ComputeStorage;

/// The MemoryHandle trait is an abstract way to refer to some memory segment.
/// It should not contain actual references to data.
///
/// It is responsible for determining if the memory segment can be mutated,
/// for instance by keeping track of a reference count
pub trait MemoryHandle: Clone + Send {
    /// Checks if the underlying memory can be safely mutated.
    fn can_mut(&self) -> bool;
}

/// The MemoryManagement trait encapsulates strategies for (de)allocating memory.
/// It is bound to the ComputeStorage trait, which does the actual (de)allocations.
///
/// The MemoryManagement can only reserve memory space or get the resource located at a space.
/// Modification of the resource data should be done directly on the resource.
pub trait MemoryManagement<Storage: ComputeStorage>: Send {
    /// The associated type Handle must implement MemoryHandle
    type Handle: MemoryHandle;

    /// Returns the resource from the storage at the specified handle
    fn get(&mut self, handle: &Self::Handle) -> Storage::Resource;

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    fn reserve(&mut self, size: usize) -> Self::Handle;
}
