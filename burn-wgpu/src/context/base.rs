use burn_common::id::IdGenerator;
use spin::Mutex;
use std::{any::TypeId, borrow::Cow, collections::HashMap, sync::Arc};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, ComputePipeline, DeviceDescriptor, DeviceType, ShaderModuleDescriptor,
};

use crate::{
    context::{client::ContextClient, server::ContextServer},
    kernel::{DynamicKernelGenerator, StaticKernelGenerator},
    GraphicsApi, WgpuDevice,
};

/// The context is the basic struct that allows to execute GPU kernel on devices.
///
/// You can access a context for a [wgpu device](WGPUDevice) using [get_context](crate::pool::get_context).
#[derive(Debug)]
pub struct Context {
    id: String,
    device_wgpu: Arc<wgpu::Device>,
    cache: Mutex<HashMap<Key, Arc<ComputePipeline>>>,
    client: ContextClient,
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
        let limits = wgpu::Limits {
            max_compute_workgroup_storage_size: 1024,
            ..wgpu::Limits::default()
        };

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
        let client = ContextServer::start(device.clone(), queue);

        Self {
            id: IdGenerator::generate(),
            device_wgpu: device,
            device: device_wgpu,
            client,
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

        self.client.compute(bind_group, pipeline, work_group)
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
    ///
    /// Wait for registered may be useful if you want to allow inplace operations on the created
    /// buffer. Otherwise, the strong count of the buffer might not be 1 when registering a new
    /// operation, which makes the buffer readonly.
    pub fn create_buffer_with_data(&self, data: &[u8], wait_for_registered: bool) -> Arc<Buffer> {
        let buffer_src = Arc::new(self.device_wgpu.create_buffer_init(&BufferInitDescriptor {
            label: Some("Buffer Src"),
            contents: data,
            usage: wgpu::BufferUsages::COPY_SRC,
        }));

        let buffer_dest = self.create_buffer(buffer_src.size() as usize);

        self.client
            .copy_buffer(buffer_src, buffer_dest, wait_for_registered)
    }

    /// Copy buffer to buffer.
    pub fn copy_buffer(&self, buffer_src: Arc<Buffer>, wait_for_registered: bool) -> Arc<Buffer> {
        let buffer_dest = self.create_buffer(buffer_src.size() as usize);

        self.client
            .copy_buffer(buffer_src, buffer_dest, wait_for_registered)
    }

    /// Read a buffer from the GPU and return its content as bytes.
    pub fn read_buffer(&self, buffer: Arc<Buffer>) -> Vec<u8> {
        self.client.read(buffer)
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
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
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
