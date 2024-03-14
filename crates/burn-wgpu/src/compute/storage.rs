use crate::compute::{BindingResource, Buffer, BufferBinding, BufferDescriptor, Device, WebGPUApi};
use burn_compute::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use hashbrown::HashMap;
use std::{num::NonZeroU64, sync::Arc};

/// Buffer storage for wgpu.
pub struct WgpuStorage<W: WebGPUApi> {
    memory: HashMap<StorageId, Arc<W::Buffer>>,
    deallocations: Vec<StorageId>,
    device: Arc<W::Device>,
}

impl<W> core::fmt::Debug for WgpuStorage<W>
where
    W: WebGPUApi,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuResource<W: WebGPUApi> {
    /// The wgpu buffer.
    pub buffer: Arc<W::Buffer>,
    /// How the resource is used.
    pub kind: WgpuResourceKind,
}

impl<W> WgpuResource<W>
where
    W: WebGPUApi,
{
    /// Return the binding view of the buffer.
    pub fn as_binding(&self) -> BindingResource<'_, W::Buffer> {
        let binding = match &self.kind {
            WgpuResourceKind::Full => self.buffer.as_entire_buffer_binding(),
            WgpuResourceKind::Slice { offset, size } => BufferBinding::<'_> {
                buffer: self.buffer.as_ref(),
                offset: *offset,
                size: Some(*size),
            },
        };
        BindingResource::Buffer(binding)
    }

    /// Return the buffer size.
    pub fn size(&self) -> u64 {
        match self.kind {
            WgpuResourceKind::Full => self.buffer.size(),
            WgpuResourceKind::Slice { offset: _, size } => size.get(),
        }
    }

    /// Return the buffer offset.
    pub fn offset(&self) -> u64 {
        match self.kind {
            WgpuResourceKind::Full => 0,
            WgpuResourceKind::Slice { offset, size: _ } => offset,
        }
    }
}

/// How the resource is used, either as a slice or fully.
#[derive(Debug)]
pub enum WgpuResourceKind {
    /// Represents an entire buffer.
    Full,
    /// A slice over a buffer.
    Slice { offset: u64, size: NonZeroU64 },
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl<W> WgpuStorage<W>
where
    W: WebGPUApi,
{
    /// Create a new storage on the given [device](WebGPUDevice).
    pub fn new(device: Arc<W::Device>) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            device,
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

impl<W> ComputeStorage for WgpuStorage<W>
where
    W: WebGPUApi,
{
    type Resource = WgpuResource<W>;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let buffer = self.memory.get(&handle.id).unwrap();

        match handle.utilization {
            StorageUtilization::Full(_) => {
                WgpuResource::new(buffer.clone(), WgpuResourceKind::Full)
            }
            StorageUtilization::Slice { offset, size } => WgpuResource::new(
                buffer.clone(),
                WgpuResourceKind::Slice {
                    offset: offset as u64,
                    size: NonZeroU64::new(size as u64).unwrap(),
                },
            ),
        }
    }

    fn alloc(&mut self, size: usize) -> StorageHandle {
        let id = StorageId::new();

        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: size as u64,
            usage: W::COPY_DST | W::STORAGE | W::COPY_SRC,
            mapped_at_creation: false,
        });

        self.memory.insert(id.clone(), buffer.into());

        StorageHandle::new(id, StorageUtilization::Full(size))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }
}
