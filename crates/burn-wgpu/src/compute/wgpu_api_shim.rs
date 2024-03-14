use crate::{
    compute::{webgpu_api::*, WgpuServer, WgpuStorage},
    create_client, GraphicsApi, RuntimeOptions, WgpuDevice,
};
use alloc::sync::Arc;
use burn_compute::{
    channel::MutexComputeChannel, client::ComputeClient, memory_management::SimpleMemoryManagement,
    ComputeRuntime,
};
use burn_jit::compute::WorkGroup;

#[derive(Debug)]
pub struct WgpuApi {}

pub struct WgpuBackend {
    backend: wgpu::Backend,
}

impl Adapter<wgpu::AdapterInfo> for wgpu::Adapter {
    fn get_info(&self) -> wgpu::AdapterInfo {
        wgpu::Adapter::get_info(self)
    }
}

impl AdapterInfo<WgpuBackend> for wgpu::AdapterInfo {
    fn backend(&self) -> WgpuBackend {
        WgpuBackend {
            backend: self.backend,
        }
    }

    fn device(&self) -> DeviceId {
        self.device
    }
}

impl core::convert::AsRef<str> for WgpuBackend {
    fn as_ref(&self) -> &str {
        wgpu::Backend::to_str(self.backend)
    }
}

impl BindGroup for wgpu::BindGroup {}

impl BindGroupLayout for wgpu::BindGroupLayout {}

impl Buffer<wgpu::Buffer, wgpu::Device> for wgpu::Buffer {
    fn as_entire_buffer_binding(&self) -> BufferBinding<'_, wgpu::Buffer> {
        let binding = wgpu::Buffer::as_entire_buffer_binding(self);
        BufferBinding {
            buffer: binding.buffer,
            offset: binding.offset,
            size: binding.size,
        }
    }

    fn destroy(&self) {
        wgpu::Buffer::destroy(self)
    }

    async fn read(&self, device: &wgpu::Device) -> Vec<u8> {
        let buffer_slice = self.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender
                .send(v)
                .expect("Unable to send buffer slice result to async channel.")
        });

        device.poll(wgpu::Maintain::Wait);

        let result = receiver.receive().await;

        if let Some(Ok(())) = result {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            self.unmap();
            result
        } else {
            panic!("Unable to read buffer {:?}", result)
        }
    }

    fn size(&self) -> u64 {
        wgpu::Buffer::size(self)
    }
}

impl CommandBuffer for wgpu::CommandBuffer {}

impl CommandEncoder<wgpu::BindGroup, wgpu::Buffer, wgpu::CommandBuffer, wgpu::ComputePipeline>
    for wgpu::CommandEncoder
{
    fn dispatch_compute_pass(
        &mut self,
        desc: &ComputePassDescriptor,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: wgpu::BindGroup,
        work_group: WorkGroup,
    ) {
        let mut compute = self.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: desc.label,
            timestamp_writes: None,
        });

        compute.set_pipeline(&pipeline);
        compute.set_bind_group(0, &bind_group, &[]);
        compute.dispatch_workgroups(work_group.x, work_group.y, work_group.z);
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: &wgpu::Buffer,
        src_offset: u64,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        size: u64,
    ) {
        wgpu::CommandEncoder::copy_buffer_to_buffer(self, src, src_offset, dst, dst_offset, size)
    }

    fn finish(self) -> wgpu::CommandBuffer {
        wgpu::CommandEncoder::finish(self)
    }
}

impl ComputePipeline<wgpu::BindGroupLayout> for wgpu::ComputePipeline {
    fn get_bind_group_layout(&self, id: u32) -> wgpu::BindGroupLayout {
        wgpu::ComputePipeline::get_bind_group_layout(self, id)
    }
}

impl
    Device<
        wgpu::BindGroup,
        wgpu::BindGroupLayout,
        wgpu::Buffer,
        wgpu::CommandEncoder,
        wgpu::ComputePipeline,
        wgpu::PipelineLayout,
        wgpu::ShaderModule,
    > for wgpu::Device
{
    fn create_bind_group(
        &self,
        desc: &BindGroupDescriptor<'_, wgpu::BindGroupLayout, wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        let entries = desc
            .entries
            .iter()
            .map(|entry| {
                let BindingResource::Buffer(resource) = &entry.resource;
                wgpu::BindGroupEntry {
                    binding: entry.binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: resource.buffer,
                        offset: resource.offset,
                        size: resource.size,
                    }),
                }
            })
            .collect::<Vec<_>>();

        wgpu::Device::create_bind_group(
            self,
            &wgpu::BindGroupDescriptor {
                label: desc.label,
                layout: desc.layout,
                entries: &entries,
            },
        )
    }

    fn create_buffer(&self, desc: &BufferDescriptor) -> wgpu::Buffer {
        wgpu::Device::create_buffer(
            self,
            &wgpu::BufferDescriptor {
                label: desc.label,
                size: desc.size,
                usage: wgpu::BufferUsages::from_bits(desc.usage).unwrap(),
                mapped_at_creation: desc.mapped_at_creation,
            },
        )
    }

    fn create_buffer_init(&self, desc: &BufferInitDescriptor) -> wgpu::Buffer {
        wgpu::util::DeviceExt::create_buffer_init(
            self,
            &wgpu::util::BufferInitDescriptor {
                label: desc.label,
                contents: desc.contents,
                usage: wgpu::BufferUsages::from_bits(desc.usage).unwrap(),
            },
        )
    }

    fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> wgpu::CommandEncoder {
        wgpu::Device::create_command_encoder(
            self,
            &wgpu::CommandEncoderDescriptor { label: desc.label },
        )
    }

    fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor<wgpu::PipelineLayout, wgpu::ShaderModule>,
    ) -> wgpu::ComputePipeline {
        wgpu::Device::create_compute_pipeline(
            self,
            &wgpu::ComputePipelineDescriptor {
                label: desc.label,
                layout: desc.layout,
                module: desc.module,
                entry_point: desc.entry_point,
            },
        )
    }

    fn create_shader_module(&self, desc: &ShaderModuleDescriptor) -> wgpu::ShaderModule {
        let source = match &desc.source {
            ShaderSource::Wgsl(source) => source.to_string(),
        };
        wgpu::Device::create_shader_module(
            self,
            wgpu::ShaderModuleDescriptor {
                label: desc.label,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            },
        )
    }
}

impl PipelineLayout for wgpu::PipelineLayout {}

impl Queue<wgpu::Buffer, wgpu::CommandBuffer> for wgpu::Queue {
    fn submit(&self, buf: Option<wgpu::CommandBuffer>) {
        wgpu::Queue::submit(self, buf);
    }

    fn write_buffer(&self, buf: &wgpu::Buffer, offset: u64, data: &[u8]) {
        wgpu::Queue::write_buffer(self, buf, offset, data)
    }
}

impl ShaderModule for wgpu::ShaderModule {}

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

type Server = WgpuServer<WgpuApi, SimpleMemoryManagement<WgpuStorage<WgpuApi>>>;

impl WebGPUApi for WgpuApi {
    type Adapter = wgpu::Adapter;
    type AdapterInfo = wgpu::AdapterInfo;
    type Backend = WgpuBackend;
    type BindGroup = wgpu::BindGroup;
    type BindGroupLayout = wgpu::BindGroupLayout;
    type Buffer = wgpu::Buffer;
    type CommandBuffer = wgpu::CommandBuffer;
    type CommandEncoder = wgpu::CommandEncoder;
    type ComputePipeline = wgpu::ComputePipeline;
    type Device = wgpu::Device;
    type PipelineLayout = wgpu::PipelineLayout;
    type Queue = wgpu::Queue;
    type ShaderModule = wgpu::ShaderModule;

    const MAP_READ: u32 = wgpu::BufferUsages::MAP_READ.bits();
    const COPY_SRC: u32 = wgpu::BufferUsages::COPY_SRC.bits();
    const COPY_DST: u32 = wgpu::BufferUsages::COPY_DST.bits();
    const STORAGE: u32 = wgpu::BufferUsages::STORAGE.bits();

    type Server = WgpuServer<Self, SimpleMemoryManagement<WgpuStorage<Self>>>;
    type Channel = MutexComputeChannel<WgpuServer<Self, SimpleMemoryManagement<WgpuStorage<Self>>>>;

    fn client<G: GraphicsApi>(device: &WgpuDevice) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            pollster::block_on(create_client::<WgpuApi, G>(
                device,
                RuntimeOptions::default(),
            ))
        })
    }

    async fn select_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
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

        (device, queue)
    }

    #[cfg(target_family = "wasm")]
    async fn select_adapter(_device: &WgpuDevice) -> Self::Adapter {
        let instance = wgpu::Instance::default();

        instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
            .await
            .unwrap()
    }

    #[cfg(not(target_family = "wasm"))]
    fn select_adapter<G: GraphicsApi>(device: &WgpuDevice) -> wgpu::Adapter {
        use wgpu::DeviceType;

        let instance = wgpu::Instance::default();
        let mut adapters_other = Vec::new();
        let mut adapters = Vec::new();

        instance
            .enumerate_adapters(G::backend().into())
            .into_iter()
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

                adapters
                    .into_iter()
                    .chain(adapters_other)
                    .for_each(|adapter| {
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

    fn device_poll(device: &Self::Device) {
        device.poll(wgpu::Maintain::Wait);
    }

    fn init_sync<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
        let device = Arc::new(device);
        let client = pollster::block_on(create_client::<Self, G>(&device, options));

        RUNTIME.register(&device, client)
    }

    async fn init_async<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
        let device = Arc::new(device);
        let client = create_client::<Self, G>(&device, options).await;

        RUNTIME.register(&device, client)
    }
}
