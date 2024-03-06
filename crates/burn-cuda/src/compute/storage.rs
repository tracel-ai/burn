use burn_compute::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};
use std::{collections::HashMap, sync::Arc};

/// Buffer storage for cuda.
pub struct CudaStorage {
    memory: HashMap<StorageId, CudaSlice<u8>>,
    deallocations: Vec<StorageId>,
    device: Arc<CudaDevice>,
}

impl core::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("CudaStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl CudaStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            device,
        }
    }

    /// Actually deallocates buffers tagged to be deallocated.
    pub fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            let _ = self.memory.remove(&id);
        }
    }
}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct CudaResource<'a> {
    /// The wgpu buffer.
    pub buffer: &'a mut CudaSlice<u8>,
    /// How the resource is used.
    pub kind: CudaResourceKind,
}

#[derive(new, Debug, Copy, Clone)]
pub struct Binding {
    ptr: *mut std::ffi::c_void,
}

unsafe impl Send for Binding {}
unsafe impl Sync for Binding {}

unsafe impl DeviceRepr for Binding {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}
impl<'a> CudaResource<'a> {
    /// Return the binding view of the buffer.
    pub fn as_binding(&self) -> Binding {
        let ptr = DeviceRepr::as_kernel_param(&self.buffer);

        let ptr = match self.kind {
            CudaResourceKind::Full { size: _ } => ptr,
            CudaResourceKind::Slice { size: _, offset } => {
                panic!("NOOOOOOOOOoo");
                ptr.wrapping_add(offset * std::mem::size_of::<u8>())
            }
        };
        Binding::new(ptr)
    }
    /// Return the binding view of the buffer.
    pub fn view(&'a self) -> cudarc::driver::CudaView<'a, u8> {
        match &self.kind {
            CudaResourceKind::Full { size } => {
                println!("Here");
                self.buffer.slice(0..*size)
            }
            CudaResourceKind::Slice { size, offset } => {
                panic!("NOOOOOOOOOoo view");
                self.buffer.slice(*offset..*size + *offset)
            },
        }
    }

    /// Return the buffer size.
    pub fn size(&self) -> u64 {
        match self.kind {
            CudaResourceKind::Full { size } => size as u64,
            CudaResourceKind::Slice { size, offset: _ } => size as u64,
        }
    }

    /// Return the buffer offset.
    pub fn offset(&self) -> u64 {
        match self.kind {
            CudaResourceKind::Full { size: _ } => 0,
            CudaResourceKind::Slice { size: _, offset } => offset as u64,
        }
    }
}

/// How the resource is used, either as a slice or fully.
#[derive(Debug)]
pub enum CudaResourceKind {
    /// Represents an entire buffer.
    Full { size: usize },
    /// A slice over a buffer.
    Slice { size: usize, offset: usize },
}

impl ComputeStorage for CudaStorage {
    type Resource<'a> = CudaResource<'a>;

    fn get<'a>(&'a mut self, handle: &StorageHandle) -> Self::Resource<'a> {
        let buffer = self.memory.get_mut(&handle.id).unwrap();

        match handle.utilization {
            StorageUtilization::Full(size) => {
                CudaResource::new(buffer, CudaResourceKind::Full { size })
            }
            StorageUtilization::Slice(offset, size) => {
                CudaResource::new(buffer, CudaResourceKind::Slice { size, offset })
            }
        }
    }

    fn alloc(&mut self, size: usize) -> StorageHandle {
        let id = StorageId::new();
        let buffer = self.device.alloc_zeros::<u8>(size).unwrap();

        self.memory.insert(id.clone(), buffer);

        StorageHandle::new(id, StorageUtilization::Full(size))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }
}
