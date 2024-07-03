use crate::storage::ComputeStorage;

/// The managed tensor buffer handle that points to some memory segment.
/// It should not contain actual data.
pub trait MemoryHandle<Binding>: Clone + Send + Sync + core::fmt::Debug {
    /// Checks if the underlying memory can be safely mutated.
    fn can_mut(&self) -> bool;
    /// Get the binding associated to the current handle.
    fn binding(self) -> Binding;
}

/// Binding to a [memory handle](MemoryHandle).
pub trait MemoryBinding: Clone + Send + Sync + core::fmt::Debug {}

/// The MemoryManagement trait encapsulates strategies for (de)allocating memory.
/// It is bound to the ComputeStorage trait, which does the actual (de)allocations.
///
/// The MemoryManagement can only reserve memory space or get the resource located at a space.
/// Modification of the resource data should be done directly on the resource.
pub trait MemoryManagement<Storage: ComputeStorage>: Send + core::fmt::Debug {
    /// The associated type that must implement [MemoryHandle].
    type Handle: MemoryHandle<Self::Binding>;
    /// The associated type that must implement [MemoryBinding]
    type Binding: MemoryBinding;

    /// Returns the resource from the storage at the specified handle
    fn get(&mut self, binding: Self::Binding) -> Storage::Resource;

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    fn reserve<Sync: FnOnce()>(&mut self, size: usize, sync: Sync) -> Self::Handle;

    /// Bypass the memory allocation algorithm to allocate data directly.
    ///
    /// # Notes
    ///
    /// Can be useful for servers that want specific control over memory.
    fn alloc<Sync: FnOnce()>(&mut self, size: usize, sync: Sync) -> Self::Handle;

    /// Bypass the memory allocation algorithm to deallocate data directly.
    ///
    /// # Notes
    ///
    /// Can be useful for servers that want specific control over memory.
    fn dealloc(&mut self, binding: Self::Binding);

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
