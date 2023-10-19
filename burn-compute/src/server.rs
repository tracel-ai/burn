use crate::{
    memory_management::{MemoryHandle, MemoryManagement},
    storage::ComputeStorage,
};
use alloc::vec::Vec;
use burn_common::reader::Reader;

/// The compute server is responsible for handling resources and computations over resources.
///
/// Everything in the server is mutable, therefore it should be solely accessed through the
/// [compute channel](crate::channel::ComputeChannel) for thread safety.
pub trait ComputeServer: Send + core::fmt::Debug
where
    Self: Sized,
{
    /// The kernel type defines the computation algorithms.
    type Kernel: Send + Clone;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;
    /// The [memory management](MemoryManagement) type defines strategies for allocation in the [storage](ComputeStorage) type.
    type MemoryManagement: MemoryManagement<Self::Storage>;

    /// Given a handle, returns the owned resource as bytes.
    fn read(&mut self, handle: &Handle<Self>) -> Reader<Vec<u8>>;

    /// Given a resource as bytes, stores it and returns the memory handle.
    fn create(&mut self, data: &[u8]) -> Handle<Self>;

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    fn empty(&mut self, size: usize) -> Handle<Self>;

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    fn execute(&mut self, kernel: Self::Kernel, handles: &[&Handle<Self>]);

    /// Wait for the completion of every task in the server.
    fn sync(&mut self);
}

/// Server handle containing the [memory handle](MemoryManagement::Handle).
#[derive(new, Debug)]
pub struct Handle<Server: ComputeServer> {
    /// Handle for the memory in use.
    pub memory: <Server::MemoryManagement as MemoryManagement<Server::Storage>>::Handle,
}

impl<Server: ComputeServer> Handle<Server> {
    /// If the tensor handle can be mut with an inplace operation.
    pub fn can_mut(&self) -> bool {
        self.memory.can_mut()
    }
}

impl<Server: ComputeServer> Clone for Handle<Server> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
        }
    }
}
