use crate::{
    memory_management::{self, MemoryManagement},
    storage::ComputeStorage,
    tune::AutotuneKey,
};
use alloc::vec::Vec;
use burn_common::reader::Reader;
use core::fmt::Debug;

/// The compute server is responsible for handling resources and computations over resources.
///
/// Everything in the server is mutable, therefore it should be solely accessed through the
/// [compute channel](crate::channel::ComputeChannel) for thread safety.
pub trait ComputeServer: Send + core::fmt::Debug
where
    Self: Sized,
{
    /// The kernel type defines the computation algorithms.
    type Kernel: Send;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;
    /// The [memory management](MemoryManagement) type defines strategies for allocation in the [storage](ComputeStorage) type.
    type MemoryManagement: MemoryManagement<Self::Storage>;
    /// The key used to cache operations used on specific inputs in autotune
    type AutotuneKey: AutotuneKey;

    /// Given a handle, returns the owned resource as bytes.
    fn read(&mut self, handle: ExecutionBufferHandle<Self>) -> Reader<Vec<u8>>;

    /// Given a resource as bytes, stores it and returns the memory handle.
    fn create(&mut self, data: &[u8]) -> TensorBufferHandle<Self>;

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    fn empty(&mut self, size: usize) -> TensorBufferHandle<Self>;

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    fn execute(&mut self, kernel: Self::Kernel, handles: Vec<ExecutionBufferHandle<Self>>);

    /// Wait for the completion of every task in the server.
    fn sync(&mut self);
}

/// Server handle containing the [memory handle](MemoryManagement::Handle).
#[derive(new, Debug)]
pub struct TensorBufferHandle<Server: ComputeServer> {
    /// Handle for the memory in use.
    pub memory: <Server::MemoryManagement as MemoryManagement<Server::Storage>>::TensorBufHandle,
}

/// Server handle containing the [memory handle](MemoryManagement::Handle).
#[derive(new)]
pub struct ExecutionBufferHandle<Server: ComputeServer> {
    /// Handle for the memory in use.
    pub memory: <Server::MemoryManagement as MemoryManagement<Server::Storage>>::BufHandle,
}

impl<Server: ComputeServer> TensorBufferHandle<Server> {
    /// If the tensor handle can be mut with an inplace operation.
    pub fn can_mut(&self) -> bool {
        memory_management::MemoryTensorBufHandle::can_mut(&self.memory)
    }
}

impl<Server: ComputeServer> TensorBufferHandle<Server> {
    /// Server handle id.
    pub fn execution(&self) -> ExecutionBufferHandle<Server> {
        ExecutionBufferHandle {
            memory: memory_management::MemoryTensorBufHandle::enqueue(&self.memory),
        }
    }
}

impl<Server: ComputeServer> Clone for TensorBufferHandle<Server> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
        }
    }
}

impl<Server: ComputeServer> Clone for ExecutionBufferHandle<Server> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
        }
    }
}
