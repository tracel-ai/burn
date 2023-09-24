use crate::storage::ComputeStorage;

/// The MemoryHandle trait is an abstract way to refer to some memory segment.
/// It should not contain actual references to data.
///
/// It is responsible for determining if the memory segment can be mutated,
/// for instance by keeping track of a reference count
pub trait MemoryHandle: Clone + Send + core::fmt::Debug {
    /// Checks if the underlying memory can be safely mutated.
    fn can_mut(&self) -> bool;
}

/// The MemoryManagement trait encapsulates strategies for (de)allocating memory.
/// It is bound to the ComputeStorage trait, which does the actual (de)allocations.
///
/// The MemoryManagement can only reserve memory space or get the resource located at a space.
/// Modification of the resource data should be done directly on the resource.
pub trait MemoryManagement<Storage: ComputeStorage>: Send + core::fmt::Debug {
    /// The associated type Handle must implement MemoryHandle
    type Handle: MemoryHandle;

    /// Returns the resource from the storage at the specified handle
    fn get(&mut self, handle: &Self::Handle) -> Storage::Resource;

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    fn reserve(&mut self, size: usize) -> Self::Handle;

    /// Bypass the memory allocation algorithm to allocate data directly.
    ///
    /// # Notes
    ///
    /// Can be useful for servers that want specific control over memory.
    fn alloc(&mut self, size: usize) -> Self::Handle;

    /// Bypass the memory allocation algorithm to deallocate data directly.
    ///
    /// # Notes
    ///
    /// Can be useful for servers that want specific control over memory.
    fn dealloc(&mut self, handle: &Self::Handle);

    /// Fetch the storage used by the memory manager.
    ///
    /// # Notes
    ///
    /// The storage should probably not be used for allocations since the handles won't be
    /// compatible with the ones provided by the current trait. Prefer using the
    /// [alloc](MemoryManagement::alloc) and [dealloc](MemoryManagement::dealloc) functions.
    ///
    /// This is useful if you need to time the deallocations based on async computation, or to
    /// change the mode of storage for different reasons.
    fn storage(&mut self) -> &mut Storage;
}
