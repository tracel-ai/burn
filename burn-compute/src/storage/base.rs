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

impl StorageHandle {
    pub fn size(&self) -> usize {
        match self.utilization {
            StorageUtilization::Full(size) => size,
            StorageUtilization::Slice(_, size) => size,
        }
    }
}

pub trait ComputeStorage {
    type Resource;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;
    fn alloc(&mut self, size: usize) -> StorageHandle;
    fn dealloc(&mut self, handle: &StorageHandle);
}
