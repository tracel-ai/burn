use crate::{ComputeStorage, MemoryManagement};
use alloc::vec::Vec;

pub type ServerResource<Server> =
    <<Server as ComputeServer>::MemoryManagement as MemoryManagement<
        <Server as ComputeServer>::Storage,
    >>::Resource;

pub trait ComputeServer {
    type KernelDescription;
    type Storage: ComputeStorage;
    type MemoryManagement: MemoryManagement<Self::Storage>;

    fn read(&mut self, resource_description: &ServerResource<Self>) -> Vec<u8>;

    fn create(&mut self, resource: Vec<u8>) -> ServerResource<Self>;

    fn empty(&mut self, size: usize) -> ServerResource<Self>;

    fn execute(
        &mut self,
        kernel_description: Self::KernelDescription,
        resource_descriptions: &[&ServerResource<Self>],
    );

    fn sync(&self);
}
