use crate::ComputeStorage;

pub trait MemoryHandle: Clone {
    fn can_mut(&self) -> bool;
}

pub trait MemoryManagement<Storage: ComputeStorage> {
    type Handle: MemoryHandle;

    fn get(&mut self, handle: &Self::Handle) -> Storage::Resource;
    fn reserve(&mut self, size: usize) -> Self::Handle;
}
