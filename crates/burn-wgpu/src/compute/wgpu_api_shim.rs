use crate::{GraphicsApi, WgpuDevice};

pub type WebGPUAdapter = wgpu::Adapter;
pub type WebGPUAdapterInfo = wgpu::AdapterInfo;
pub type WebGPUBindGroup = wgpu::BindGroup;
pub type WebGPUBindGroupDescriptor<'a> = wgpu::BindGroupDescriptor<'a>;
pub type WebGPUBindGroupEntry<'a> = wgpu::BindGroupEntry<'a>;
pub type WebGPUBindingResource<'a> = wgpu::BindingResource<'a>;
pub type WebGPUBuffer = wgpu::Buffer;
pub type WebGPUBufferAddress = wgpu::BufferAddress;
pub type WebGPUBufferBinding<'a> = wgpu::BufferBinding<'a>;
pub type WebGPUBufferSize = wgpu::BufferSize;
pub type WebGPUBufferDescriptor<'a> = wgpu::BufferDescriptor<'a>;

pub type WebGPUBufferUsages = wgpu::BufferUsages;
pub const MAP_READ: WebGPUBufferUsages = wgpu::BufferUsages::MAP_READ;
pub const COPY_SRC: WebGPUBufferUsages = wgpu::BufferUsages::COPY_SRC;
pub const COPY_DST: WebGPUBufferUsages = wgpu::BufferUsages::COPY_DST;
pub const STORAGE: WebGPUBufferUsages = wgpu::BufferUsages::STORAGE;

pub type WebGPUCommandEncoder = wgpu::CommandEncoder;
pub type WebGPUCommandEncoderDescriptor<'a> = wgpu::CommandEncoderDescriptor<'a>;
pub type WebGPUComputePassDescriptor<'a> = wgpu::ComputePassDescriptor<'a>;
pub type WebGPUComputePipeline = wgpu::ComputePipeline;
pub type WebGPUComputePipelineDescriptor<'a> = wgpu::ComputePipelineDescriptor<'a>;
pub type WebGPUDevice = wgpu::Device;
pub type WebGPUQueue = wgpu::Queue;
pub type WebGPUShaderModuleDescriptor<'a> = wgpu::ShaderModuleDescriptor<'a>;
pub type WebGPUShaderSource<'a> = wgpu::ShaderSource<'a>;

pub async fn webgpu_select_device(adapter: &WebGPUAdapter) -> (WebGPUDevice, WebGPUQueue) {
    let limits = adapter.limits();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
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

    (device, queue)
}

#[cfg(target_family = "wasm")]
async fn webgpu_select_adapter(_device: &WgpuDevice) -> WebGPUAdapter {
    let instance = wgpu::Instance::default();

    instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
        .await
        .unwrap()
}

#[cfg(not(target_family = "wasm"))]
pub fn webgpu_select_adapter<G: GraphicsApi>(device: &WgpuDevice) -> WebGPUAdapter {
    use wgpu::DeviceType;

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

pub fn webgpu_read_buffer(buffer: &WebGPUBuffer, device: &WebGPUDevice) -> Vec<u8> {
    pollster::block_on(webgpu_read_buffer_async(buffer, device))
}

async fn webgpu_read_buffer_async(buffer: &WebGPUBuffer, device: &WebGPUDevice) -> Vec<u8> {
    let buffer_slice = buffer.slice(..);
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
        buffer.unmap();
        result
    } else {
        panic!("Unable to read buffer {:?}", result)
    }
}

pub fn webgpu_device_poll(device: &WebGPUDevice) {
    device.poll(wgpu::Maintain::Wait);
}
