use std::{
    num::NonZeroU64,
    sync::{Arc, Mutex},
};

use crate::context::client::ContextClient;

// struct BufferBlock {
//     buffer: wgpu::Buffer,
// }

#[derive(Debug, Clone)]
pub(crate) enum GpuBufferSrc {
    FullBuffer,
    Slice(wgpu::BufferAddress, wgpu::BufferAddress),
}

#[derive(Debug, Clone)]
pub(crate) struct GpuBuffer {
    desc: GpuBufferSrc,
    buffer: Arc<wgpu::Buffer>,
}

impl GpuBuffer {
    /// Return the binding view of the buffer.
    pub fn as_binding(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Buffer(self.as_buffer_binding())
    }

    /// Return the binding view of the buffer.
    pub fn as_buffer_binding(&self) -> wgpu::BufferBinding {
        match &self.desc {
            GpuBufferSrc::FullBuffer => self.buffer.as_entire_buffer_binding(),
            GpuBufferSrc::Slice(offs, size) => wgpu::BufferBinding {
                buffer: &self.buffer,
                offset: *offs,
                size: NonZeroU64::new(*size),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum BufferSrc {
    Resident(GpuBuffer),
    NonResident(Option<Arc<[u8]>>),
}

#[derive(Debug)]
struct BufferInner {
    /// The description of where the buffer is currently located.
    src: BufferSrc,
    /// The last time the buffer was used
    usage: std::time::Instant,
}

#[derive(Debug, Clone)]
pub(crate) struct BufferDesc {
    /// Debug label of a buffer. This will show up in graphics debuggers for easy identification.
    pub label: Option<String>,
    /// Size of a buffer, in bytes.
    pub size: wgpu::BufferAddress,
    /// Usages of a buffer. If the buffer is used in any way that isn't specified here, the operation
    /// will panic.
    pub usage: wgpu::BufferUsages,
}

impl From<&wgpu::BufferDescriptor<'_>> for BufferDesc {
    fn from(desc: &wgpu::BufferDescriptor) -> Self {
        Self {
            label: desc.label.map(|s| s.to_owned()),
            size: desc.size,
            usage: desc.usage,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Buffer {
    desc: BufferDesc,
    inner: Arc<Mutex<BufferInner>>,
}

impl Buffer {
    /// Create a new buffer with data located in system memory.
    pub(crate) fn new_nonresident(desc: impl Into<BufferDesc>, contents: Option<&[u8]>) -> Self {
        Self {
            desc: desc.into(),
            inner: Arc::new(Mutex::new(BufferInner {
                src: BufferSrc::NonResident(contents.map(Into::into)),
                usage: std::time::Instant::now(),
            })),
        }
    }

    pub(crate) fn new_resident(desc: impl Into<BufferDesc>, buffer: Arc<wgpu::Buffer>) -> Self {
        let desc = desc.into();

        Self {
            desc: desc,
            inner: Arc::new(Mutex::new(BufferInner {
                src: BufferSrc::Resident(GpuBuffer {
                    desc: GpuBufferSrc::FullBuffer,
                    buffer: buffer,
                }),
                usage: std::time::Instant::now(),
            })),
        }
    }

    pub(crate) fn size(&self) -> u64 {
        self.desc.size
    }

    pub(crate) fn is_resident(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        matches!(inner.src, BufferSrc::Resident(_))
    }

    pub(crate) fn mark_used(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.usage = std::time::Instant::now();
    }

    pub(crate) fn last_used(&self) -> std::time::Instant {
        let inner = self.inner.lock().unwrap();
        inner.usage
    }

    pub(crate) fn read(&self, ctx: impl ContextClient, buf: &mut Vec<u8>) {
        let inner = self.inner.lock().unwrap();

        match &inner.src {
            BufferSrc::NonResident(contents) => {
                if let Some(contents) = contents {
                    buf.extend_from_slice(contents)
                }
            }
            BufferSrc::Resident(gbuffer) => {
                let bytes = ctx.read_buffer(gbuffer.buffer.clone());
                buf.extend_from_slice(&bytes);
            }
        }
    }

    /// Make a buffer resident on the GPU.
    pub(crate) fn make_resident(&self, device: &wgpu::Device) -> Result<GpuBuffer, wgpu::Error> {
        let mut inner = self.inner.lock().unwrap();

        match &inner.src {
            BufferSrc::NonResident(contents) => {
                // NOTE: Out-of-memory errors are treated as validation errors for some reason.
                device.push_error_scope(wgpu::ErrorFilter::Validation);

                assert_eq!(self.desc.size % wgpu::COPY_BUFFER_ALIGNMENT, 0);

                // Due to a bug in the `wgpu` crate, we cannot use `create_buffer_init` as it does
                // not check for an error scope before attempting to map the buffer.
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: self.desc.label.as_ref().map(|s| s.as_str()),
                    size: self.desc.size,
                    usage: self.desc.usage,
                    mapped_at_creation: contents.is_some(),
                });

                // Check to make sure the `create_buffer` operation actually succeeded. If it
                // did not, then the above `buffer` is invalid (in true C style).
                if let Some(err) = futures::executor::block_on(device.pop_error_scope()) {
                    return Err(err);
                }

                if let Some(contents) = contents {
                    buffer.slice(..).get_mapped_range_mut()[..contents.len()]
                        .copy_from_slice(contents);
                    buffer.unmap();
                }

                let buffer = GpuBuffer {
                    desc: GpuBufferSrc::FullBuffer,
                    buffer: Arc::new(buffer),
                };

                inner.src = BufferSrc::Resident(buffer.clone());

                Ok(buffer.clone())
            }
            // No-op.
            BufferSrc::Resident(gbuffer) => Ok(gbuffer.clone()),
        }
    }

    /// Evict a buffer from GPU memory to system memory.
    pub(crate) fn evict(&self, context: impl ContextClient) -> bool {
        let mut inner = self.inner.lock().unwrap();

        match &inner.src {
            BufferSrc::Resident(buff) => {
                let bytes = context.read_buffer(buff.buffer.clone());
                inner.src = BufferSrc::NonResident(Some(bytes.as_slice().into()));
                true
            }
            // No-op.
            _ => false,
        }
    }
}
