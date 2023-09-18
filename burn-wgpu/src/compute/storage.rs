use burn_compute::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use hashbrown::HashMap;
use std::{num::NonZeroU64, sync::Arc};

pub struct WgpuStorage {
    memory: HashMap<StorageId, Arc<wgpu::Buffer>>,
    device: Arc<wgpu::Device>,
}

#[derive(new, Debug)]
pub struct WgpuResource {
    pub buffer: Arc<wgpu::Buffer>,
    pub kind: WgpuResourceKind,
}

impl WgpuResource {
    /// Return the binding view of the buffer.
    pub fn as_binding(&self) -> wgpu::BindingResource {
        let binding = match &self.kind {
            WgpuResourceKind::Full => self.buffer.as_entire_buffer_binding(),
            WgpuResourceKind::Slice(offs, size) => wgpu::BufferBinding {
                buffer: &self.buffer,
                offset: *offs,
                size: Some(*size),
            },
        };
        wgpu::BindingResource::Buffer(binding)
    }

    /// Return the buffer size.
    pub fn size(&self) -> u64 {
        match self.kind {
            WgpuResourceKind::Full => self.buffer.size(),
            WgpuResourceKind::Slice(_, size) => size.get(),
        }
    }

    /// Return the buffer offset.
    pub fn offset(&self) -> u64 {
        match self.kind {
            WgpuResourceKind::Full => 0,
            WgpuResourceKind::Slice(offset, _) => offset,
        }
    }
}

#[derive(Debug)]
pub enum WgpuResourceKind {
    /// Represents an entire buffer.
    Full,
    /// A slice over a buffer.
    Slice(wgpu::BufferAddress, wgpu::BufferSize),
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            memory: HashMap::new(),
            device,
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let buffer = self.memory.get(&handle.id).unwrap();

        match handle.utilization {
            StorageUtilization::Full(_) => {
                WgpuResource::new(buffer.clone(), WgpuResourceKind::Full)
            }
            StorageUtilization::Slice(offset, size) => WgpuResource::new(
                buffer.clone(),
                WgpuResourceKind::Slice(offset as u64, NonZeroU64::new(size as u64).unwrap()),
            ),
        }
    }

    fn alloc(&mut self, size: usize) -> StorageHandle {
        let id = StorageId::new();
        let buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.memory.insert(id.clone(), buffer);

        StorageHandle::new(id, StorageUtilization::Full(size))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.memory.get(&id).unwrap().destroy();
        let _ = self.memory.remove(&id);
    }
}
