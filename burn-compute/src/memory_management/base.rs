use crate::ComputeStorage;

pub trait MemoryHandle: Clone {
    fn can_mut(&self) -> bool;
}

pub trait MemoryManagement<Storage: ComputeStorage> {
    type Handle: MemoryHandle;

    fn get(&mut self, ressource: &Self::Handle) -> Storage::StorageResource;
    fn reserve(&mut self, size: usize) -> Self::Handle;
}
