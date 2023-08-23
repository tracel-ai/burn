use super::client::ContextClient;
use crate::{
    context::server::ContextServer,
    kernel::{DynamicKernel, StaticKernel},
    tune::Tuner,
    GraphicsApi, WgpuDevice,
};
use burn_common::id::IdGenerator;
use spin::Mutex;
use std::{any::TypeId, borrow::Cow, collections::HashMap, sync::Arc};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, ComputePipeline, DeviceDescriptor, DeviceType, ShaderModuleDescriptor,
};

#[cfg(feature = "async")]
pub type ContextClientImpl = super::client::AsyncContextClient;
#[cfg(not(feature = "async"))]
pub type ContextClientImpl = super::client::SyncContextClient;

#[cfg(feature = "async")]
pub type ContextServerImpl = super::server::AsyncContextServer;
#[cfg(not(feature = "async"))]
pub type ContextServerImpl = super::server::SyncContextServer;

/// The context is the basic struct that allows to execute GPU kernel on devices.
///
/// You can access a context for a [wgpu device](WGPUDevice) using [get_context](crate::pool::get_context).
#[derive(Debug)]
pub struct Context {
    id: String,
    device_wgpu: Arc<wgpu::Device>,
    cache: Mutex<HashMap<TemplateKey, Arc<ComputePipeline>>>,
    client: ContextClientImpl,
    pub(crate) tuner: Tuner,
    pub(crate) device: WgpuDevice,
    pub(crate) info: wgpu::AdapterInfo,
}

#[derive(Debug, Hash, PartialOrd, PartialEq, Eq)]
enum TemplateKey {
    Static(TypeId),
    Dynamic(String),
}

#[derive(new, Clone, Debug)]
pub struct WorkGroup {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkGroup {
    pub fn num_invocations(&self) -> usize {
        (self.x * self.y * self.z) as usize
    }
}

impl Context {
    /// Create a new context where computing tasks will be executed on the given
    /// [device](WgpuDevice).
    pub(crate) fn new<G: GraphicsApi>(device: &WgpuDevice) -> Self {
        let (device_wgpu, queue, info) = pollster::block_on(select_device::<G>(device));
        let device = device.clone();
        let device_wgpu = Arc::new(device_wgpu);
        let client = ContextServerImpl::start(device_wgpu.clone(), queue);

        Self {
            id: IdGenerator::generate(),
            device_wgpu,
            device,
            client,
            cache: Mutex::new(HashMap::new()),
            tuner: Tuner::new(),
            info,
        }
    }

    /// Wait for all computation to be executed.
    ///
    /// Useful for benchmarks.
    pub fn sync(&self) {
        self.client.sync();
    }

    /// Execute a kernel using the provided buffers.
    ///
    /// # Notes
    ///
    /// This function isn't safe, buffer can be mutated by the GPU. The users must ensure that a
    /// buffer can be mutated when launching a compute shaders with write access to a buffer.
    ///
    /// Buffer positions are used as bindings when launching a compute kernel.
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

        self.client
            .register_compute(bind_group, pipeline, work_group)
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
        self.create_buffer_with_data_options(data, false)
    }

    /// Create a new buffer initialized with the provided bytes with the option to be sync.
    ///
    /// It's important to be sync when you want to reuse the buffer using the Arc strong count for
    /// inner mutability.
    pub fn create_buffer_with_data_options(&self, data: &[u8], sync: bool) -> Arc<Buffer> {
        let buffer_src = Arc::new(self.device_wgpu.create_buffer_init(&BufferInitDescriptor {
            label: Some("Buffer Src"),
            contents: data,
            usage: wgpu::BufferUsages::COPY_SRC,
        }));

        let buffer_dest = self.create_buffer(buffer_src.size() as usize);

        self.client.copy_buffer(buffer_src, buffer_dest, sync)
    }

    /// Copy buffer to buffer.
    ///
    /// Wait for registered may be useful if you want to allow inplace operations on the created
    /// buffer. Otherwise, the strong count of the buffer might not be 1 when registering a new
    /// operation, which makes the buffer readonly.
    pub fn copy_buffer(&self, buffer_src: Arc<Buffer>, wait_for_registered: bool) -> Arc<Buffer> {
        let buffer_dest = self.create_buffer(buffer_src.size() as usize);

        self.client
            .copy_buffer(buffer_src, buffer_dest, wait_for_registered)
    }

    /// Read a buffer from the GPU and return its content as bytes.
    pub fn read_buffer(&self, buffer: Arc<Buffer>) -> Vec<u8> {
        self.client.read_buffer(buffer)
    }

    /// Compile a kernel template if not present in the cache.
    pub fn compile_static<K: StaticKernel>(&self) -> Arc<ComputePipeline> {
        let mut cache = self.cache.lock();
        let template_id = TemplateKey::Static(TypeId::of::<K>());

        if let Some(module) = cache.get(&template_id) {
            return module.clone();
        }

        let source = K::source_template();
        let pipeline = self.compile_source(&source.complete());

        cache.insert(template_id, pipeline.clone());
        pipeline
    }

    /// Compile a dynamic template if not present in the cache.
    pub fn compile_dynamic<K: DynamicKernel>(&self, kernel: K) -> Arc<ComputePipeline> {
        let mut cache = self.cache.lock();
        let template_id = TemplateKey::Dynamic(kernel.id());

        if let Some(module) = cache.get(&template_id) {
            return module.clone();
        }

        let source = kernel.source_template();
        let pipeline = self.compile_source(&source.complete());

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

async fn select_device<G: GraphicsApi>(
    device: &WgpuDevice,
) -> (wgpu::Device, wgpu::Queue, wgpu::AdapterInfo) {
    let adapter = select_adapter::<G>(device);
    let limits = adapter.limits();

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits,
            },
            None,
        )
        .await
        .map_err(|err| {
            format!(
                "Unable to request the device with the adapter {:?}, err {:?}",
                adapter.get_info(),
                err
            )
        })
        .unwrap();

    (device, queue, adapter.get_info())
}

fn select_adapter<G: GraphicsApi>(device: &WgpuDevice) -> wgpu::Adapter {
    let instance = wgpu::Instance::default();

    let mut adapters_other = Vec::new();
    let mut adapters = Vec::new();

    instance
        .enumerate_adapters(G::backend().into())
        .for_each(|adapter| {
            let device_type = adapter.get_info().device_type;

            if let DeviceType::Other = device_type {
                adapters_other.push(adapter);
                return;
            }

            let is_same_type = match device {
                WgpuDevice::DiscreteGpu(_) => device_type == DeviceType::DiscreteGpu,
                WgpuDevice::IntegratedGpu(_) => device_type == DeviceType::IntegratedGpu,
                WgpuDevice::VirtualGpu(_) => device_type == DeviceType::VirtualGpu,
                WgpuDevice::Cpu => device_type == DeviceType::Cpu,
                WgpuDevice::BestAvailable => true,
            };

            if is_same_type {
                adapters.push(adapter);
            }
        });

    fn select(
        num: usize,
        error: &str,
        mut adapters: Vec<wgpu::Adapter>,
        mut adapters_other: Vec<wgpu::Adapter>,
    ) -> wgpu::Adapter {
        if adapters.len() <= num {
            if adapters_other.len() <= num {
                panic!(
                    "{}, adapters {:?}, other adapters {:?}",
                    error,
                    adapters
                        .into_iter()
                        .map(|adapter| adapter.get_info())
                        .collect::<Vec<_>>(),
                    adapters_other
                        .into_iter()
                        .map(|adapter| adapter.get_info())
                        .collect::<Vec<_>>(),
                );
            } else {
                return adapters_other.remove(num);
            }
        }

        adapters.remove(num)
    }

    let adapter = match device {
        WgpuDevice::DiscreteGpu(num) => select(
            *num,
            "No Discrete GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::IntegratedGpu(num) => select(
            *num,
            "No Integrated GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::VirtualGpu(num) => select(
            *num,
            "No Virtual GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::Cpu => select(0, "No CPU device found", adapters, adapters_other),
        WgpuDevice::BestAvailable => {
            let mut most_performant_adapter = None;
            let mut current_score = -1;

            adapters.into_iter().for_each(|adapter| {
                let info = adapter.get_info();
                let score = match info.device_type {
                    DeviceType::DiscreteGpu => 5,
                    DeviceType::Other => 4, // Let's be optimistic with the Other device, it's
                    // often a Discrete Gpu.
                    DeviceType::IntegratedGpu => 3,
                    DeviceType::VirtualGpu => 2,
                    DeviceType::Cpu => 1,
                };

                if score > current_score {
                    most_performant_adapter = Some(adapter);
                    current_score = score;
                }
            });

            if let Some(adapter) = most_performant_adapter {
                adapter
            } else {
                panic!("No adapter found for graphics API {:?}", G::default());
            }
        }
    };

    log::info!("Using adapter {:?}", adapter.get_info());

    adapter
}
