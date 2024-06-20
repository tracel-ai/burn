use burn_compute::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use hashbrown::HashMap;
use std::{num::NonZeroU64, sync::Arc};

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, Arc<wgpu::Buffer>>,
    deallocations: Vec<StorageId>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: Arc<wgpu::Buffer>,
    /// How the resource is used.
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

/// How the resource is used, either as a slice or fully.
#[derive(Debug)]
pub enum WgpuResourceKind {
    /// Represents an entire buffer.
    Full,
    /// A slice over a buffer.
    Slice(wgpu::BufferAddress, wgpu::BufferSize),
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            device,
            queue,
        }
    }

    /// Actually deallocates buffers tagged to be deallocated.
    pub fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            if let Some(buffer) = self.memory.remove(&id) {
                buffer.destroy()
            }
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
            StorageUtilization::Slice { offset, size } => WgpuResource::new(
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
        self.deallocations.push(id);
    }

    fn copy(&mut self, from: &StorageHandle, to: &StorageHandle) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let from = self.get(from);
        let to = self.get(to);

        encoder.copy_buffer_to_buffer(
            &from.buffer,
            from.offset(),
            &to.buffer,
            to.offset(),
            to.size(),
        );

        self.queue.submit(Some(encoder.finish()));
    }
}
