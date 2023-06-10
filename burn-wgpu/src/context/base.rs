use burn_common::id::IdGenerator;
use spin::Mutex;
use std::{
    any::TypeId,
    borrow::Cow,
    collections::HashMap,
    sync::{mpsc, Arc},
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, ComputePipeline, DeviceDescriptor, DeviceType, ShaderModuleDescriptor,
};

use crate::{
    context::background::ContextBackground,
    kernel::{DynamicKernelGenerator, StaticKernelGenerator},
    GraphicsApi, WgpuDevice,
};

use super::background::{BackgroundTask, ComputeTask, CopyBufferTask, ReadBufferTask};

/// The context is the basic struct that allows to execute GPU kernel on devices.
///
/// You can access a context for a [wgpu device](WGPUDevice) using [get_context](crate::pool::get_context).
#[derive(Debug)]
pub struct Context {
    id: String,
    device_wgpu: Arc<wgpu::Device>,
    cache: Mutex<HashMap<Key, Arc<ComputePipeline>>>,
    sender: mpsc::SyncSender<BackgroundTask>,
    _handle: std::thread::JoinHandle<()>,
    pub(crate) device: WgpuDevice,
}

#[derive(Debug, Hash, PartialOrd, PartialEq, Eq)]
enum Key {
    Static(TypeId),
    Dynamic(String),
}

#[derive(new, Clone, Debug)]
pub struct WorkGroup {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Context {
    pub(crate) fn new<G: GraphicsApi>(device: &WgpuDevice) -> Self {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapters = instance.enumerate_adapters(G::backend().into());
        let mut adapters = adapters
            .filter(|adapter| {
                let device_type = adapter.get_info().device_type;

                match device {
                    WgpuDevice::DiscreteGpu(_) => device_type == DeviceType::DiscreteGpu,
                    WgpuDevice::IntegratedGpu(_) => device_type == DeviceType::IntegratedGpu,
                    WgpuDevice::VirtualGpu(_) => device_type == DeviceType::VirtualGpu,
                    WgpuDevice::Cpu => device_type == DeviceType::Cpu,
                }
            })
            .collect::<Vec<_>>();

        let adapter = match device {
            WgpuDevice::DiscreteGpu(num) => {
                assert!(adapters.len() > *num, "No Discrete GPU device found");
                adapters.remove(*num)
            }
            WgpuDevice::IntegratedGpu(num) => {
                assert!(adapters.len() > *num, "No Integrated GPU device found");
                adapters.remove(*num)
            }
            WgpuDevice::VirtualGpu(num) => {
                assert!(adapters.len() > *num, "No Virtual GPU device found");
                adapters.remove(*num)
            }
            WgpuDevice::Cpu => {
                assert!(!adapters.is_empty(), "No CPU device found");
                adapters.remove(0)
            }
        };
        log::info!("Using adapter {:?}", adapter.get_info());

        let device_wgpu = device.clone();
        let mut limits = wgpu::Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                features: wgpu::Features::all_webgpu_mask(),
                limits,
            },
            None,
        ))
        .expect("Unable to request the device with the adapter");

        let device = Arc::new(device);

        let (sender, receiver) = std::sync::mpsc::sync_channel(50);

        let handle = ContextBackground::start(device.clone(), queue, receiver);

        Self {
            id: IdGenerator::generate(),
            device_wgpu: device,
            device: device_wgpu,
            sender,
            _handle: handle,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Execute a kernel using the provided buffers.
    ///
    /// # Notes
    ///
    /// This function isn't safe, buffer can be mutated by the GPU. The users must ensure that a
    /// buffer can be mutated when lauching a compute shaders with write access to a buffer.
    ///
    /// Buffer positions are used as bindings when lauching a compute kernel.
    pub fn execute(
        &self,
        work_group: WorkGroup,
        pipeline: Arc<ComputePipeline>,
        buffers: &[&Buffer],
    ) {
        let group_layout = pipeline.get_bind_group_layout(0);

        let entries = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        let bind_group = self
            .device_wgpu
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &group_layout,
                entries: &entries,
            });

        self.sender
            .send(ComputeTask::new(bind_group, pipeline, work_group).into())
            .unwrap();
    }

    /// Create a new buffer with the provided size.
    pub fn create_buffer(&self, size: usize) -> Arc<Buffer> {
        Arc::new(self.device_wgpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Create a new buffer initialized with the provided bytes.
    pub fn create_buffer_with_data(&self, data: &[u8]) -> Arc<Buffer> {
        let buffer_src = self.device_wgpu.create_buffer_init(&BufferInitDescriptor {
            label: Some("Buffer Src"),
            contents: data,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let buffer = self.create_buffer(buffer_src.size() as usize);

        self.sender
            .send(CopyBufferTask::new(Arc::new(buffer_src), buffer.clone()).into())
            .unwrap();

        buffer
    }

    /// Copy buffer to buffer.
    pub fn copy_buffer(&self, buffer: Arc<Buffer>) -> Arc<Buffer> {
        let buffer_out = self.create_buffer(buffer.size() as usize);

        self.sender
            .send(CopyBufferTask::new(buffer, buffer_out.clone()).into())
            .unwrap();
        buffer_out
    }

    /// Read a buffer from the GPU and return its content as bytes.
    pub fn buffer_to_data(&self, buffer: Arc<Buffer>) -> Vec<u8> {
        let (sender, receiver) = std::sync::mpsc::channel();

        self.sender
            .send(ReadBufferTask::new(buffer, sender).into())
            .unwrap();

        let mut iter = receiver.iter();
        if let Some(data) = iter.next() {
            return data;
        } else {
            panic!("Unable to read buffer")
        }
    }

    /// Compile a kernel template if not present in the cache.
    pub fn compile_static<K: StaticKernelGenerator>(&self) -> Arc<ComputePipeline> {
        let mut cache = self.cache.lock();
        let template_id = Key::Static(TypeId::of::<K>());

        if let Some(module) = cache.get(&template_id) {
            return module.clone();
        }

        let source = K::generate();
        let pipeline = self.compile_source(source.as_ref());

        cache.insert(template_id, pipeline.clone());
        pipeline
    }

    /// Compile a dynamic template if not present in the cache.
    pub fn compile_dynamic<K: DynamicKernelGenerator>(&self, kernel: K) -> Arc<ComputePipeline> {
        let mut cache = self.cache.lock();
        let template_id = Key::Dynamic(kernel.id());

        if let Some(module) = cache.get(&template_id) {
            return module.clone();
        }

        let source = kernel.generate();
        let pipeline = self.compile_source(source.as_ref());

        cache.insert(template_id, pipeline.clone());
        pipeline
    }

    fn compile_source(&self, source: &str) -> Arc<ComputePipeline> {
        let module = self
            .device_wgpu
            .create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source.as_ref())),
            });
        let pipeline = self
            .device_wgpu
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: "main",
            });

        Arc::new(pipeline)
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
