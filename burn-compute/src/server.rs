use crate::{ComputeStorage, MemoryManagement};
use alloc::vec::Vec;

type _Storage<Server> = <Server as ComputeServer>::Storage;
type _MemoryManagement<Server> = <Server as ComputeServer>::MemoryManagement;

pub type Handle<Server> = <_MemoryManagement<Server> as MemoryManagement<_Storage<Server>>>::Handle;

pub trait ComputeServer {
    type Kernel;
    type Storage: ComputeStorage;
    type MemoryManagement: MemoryManagement<Self::Storage>;

    fn read(&mut self, handle: &Handle<Self>) -> Vec<u8>;

    fn create(&mut self, data: Vec<u8>) -> Handle<Self>;

    fn empty(&mut self, size: usize) -> Handle<Self>;

    fn execute(&mut self, kernel: Self::Kernel, handles: &[&Handle<Self>]);

    fn sync(&self);
}
