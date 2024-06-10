use crate::{
    memory_management::{MemoryHandle, MemoryManagement},
    storage::ComputeStorage,
    tune::AutotuneKey,
};
use alloc::vec::Vec;
use burn_common::{reader::Reader, sync_type::SyncType};
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
    fn read(&mut self, binding: Binding<Self>) -> Reader<Vec<u8>>;

    /// Given a resource handle, returns the storage resource.
    fn get_resource(
        &mut self,
        binding: Binding<Self>,
    ) -> <Self::Storage as ComputeStorage>::Resource;

    /// Given a resource as bytes, stores it and returns the memory handle.
    fn create(&mut self, data: &[u8]) -> Handle<Self>;

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    fn empty(&mut self, size: usize) -> Handle<Self>;

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    fn execute(&mut self, kernel: Self::Kernel, bindings: Vec<Binding<Self>>);

    /// Wait for the completion of every task in the server.
    fn sync(&mut self, command: SyncType);
}

/// Server handle containing the [memory handle](MemoryManagement::Handle).
#[derive(new, Debug)]
pub struct Handle<Server: ComputeServer> {
    /// Memory handle.
    pub memory: <Server::MemoryManagement as MemoryManagement<Server::Storage>>::Handle,
}

/// Binding of a [tensor handle](Handle) to execute a kernel.
#[derive(new, Debug)]
pub struct Binding<Server: ComputeServer> {
    /// Memory binding.
    pub memory: <Server::MemoryManagement as MemoryManagement<Server::Storage>>::Binding,
}

impl<Server: ComputeServer> Handle<Server> {
    /// If the tensor handle can be reused inplace.
    pub fn can_mut(&self) -> bool {
        MemoryHandle::can_mut(&self.memory)
    }
}

impl<Server: ComputeServer> Handle<Server> {
    /// Convert the [handle](Handle) into a [binding](Binding).
    pub fn binding(self) -> Binding<Server> {
        Binding {
            memory: MemoryHandle::binding(self.memory),
        }
    }
}

impl<Server: ComputeServer> Clone for Handle<Server> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
        }
    }
}

impl<Server: ComputeServer> Clone for Binding<Server> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
        }
    }
}
