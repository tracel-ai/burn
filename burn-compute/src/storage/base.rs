use crate::id_type;

id_type!(StorageId);

#[derive(Clone)]
pub enum StorageUtilization {
    Full(usize),
    Slice(usize, usize),
}

#[derive(Clone)]
pub struct StorageHandle {
    pub id: StorageId,
    pub utilization: StorageUtilization,
}

pub trait ComputeStorage {
    type StorageResource;

    fn get(&mut self, handle: &StorageHandle) -> Self::StorageResource;
    fn alloc(&mut self, size: usize) -> StorageHandle;
    fn dealloc(&mut self, handle: &StorageHandle);
}
