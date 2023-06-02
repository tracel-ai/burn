use burn_common::id::IdGenerator;
use std::{
    any::TypeId,
    borrow::Cow,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, DeviceDescriptor, DeviceType, ShaderModule, ShaderModuleDescriptor,
};

use crate::{kernel::KernelGenerator, GraphicsAPI, WGPUDevice};

/// The context is the basic struct that allows to execute GPU kernel on devices.
///
/// You can access a context for a [wgpu device](WGPUDevice) using [get_context](crate::pool::get_context).
#[derive(Debug)]
pub struct Context {
    id: String,
    queue: wgpu::Queue,
    device_wgpu: wgpu::Device,
    cache: Mutex<HashMap<TypeId, Arc<ShaderModule>>>,
    pub(crate) device: WGPUDevice,
}

#[derive(new, Clone, Debug)]
pub struct WorkGroup {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Context {
    pub(crate) fn new<G: GraphicsAPI>(device: &WGPUDevice) -> Self {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapters = instance.enumerate_adapters(G::backend().into());
        let mut adapters = adapters
            .filter(|adapter| {
                let device_type = adapter.get_info().device_type;
                match device {
                    WGPUDevice::DiscreteGPU(_) => device_type == DeviceType::DiscreteGpu,
                    WGPUDevice::IntegratedGPU(_) => device_type == DeviceType::IntegratedGpu,
                    WGPUDevice::VirtualGPU(_) => device_type == DeviceType::VirtualGpu,
                    WGPUDevice::CPU => device_type == DeviceType::Cpu,
                }
            })
            .collect::<Vec<_>>();

        let adapter = match device {
            WGPUDevice::DiscreteGPU(num) => {
                assert!(adapters.len() > *num, "No Discrete GPU device found");
                adapters.remove(*num)
            }
            WGPUDevice::IntegratedGPU(num) => {
                assert!(adapters.len() > *num, "No Integrated GPU device found");
                adapters.remove(*num)
            }
            WGPUDevice::VirtualGPU(num) => {
                assert!(adapters.len() > *num, "No Virtual GPU device found");
                adapters.remove(*num)
            }
            WGPUDevice::CPU => {
                assert!(!adapters.is_empty(), "No CPU device found");
                adapters.remove(0)
            }
        };

        let device_wgpu = device.clone();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .expect("Unable to request the device with the adapter");

        Self {
            id: IdGenerator::generate(),
            queue,
            device_wgpu: device,
            device: device_wgpu,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Create a new buffer with the provided size.
    pub fn create_buffer(&self, size: usize) -> Buffer {
        self.device_wgpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a new buffer initialized with the provided bytes.
    pub fn create_buffer_with_data(&self, data: &[u8]) -> Buffer {
        let buffer_src = self.device_wgpu.create_buffer_init(&BufferInitDescriptor {
            label: Some("Buffer Src"),
            contents: data,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let buffer = self.create_buffer(buffer_src.size() as usize);

        // Create a command encoder
        let mut encoder =
            self.device_wgpu
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Command Encoder"),
                });

        // Copy data from the staging buffer to the target buffer
        encoder.copy_buffer_to_buffer(&buffer_src, 0, &buffer, 0, buffer_src.size());

        // Submit the command encoder to the queue
        self.queue.submit(std::iter::once(encoder.finish()));

        buffer
    }

    /// Read a buffer from the GPU and return its content as bytes.
    pub fn buffer_to_data(&self, buffer: &Buffer) -> Vec<u8> {
        let size = buffer.size();

        let buffer_dest = self.device_wgpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create a command encoder
        let mut encoder =
            self.device_wgpu
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Command Encoder"),
                });

        encoder.copy_buffer_to_buffer(buffer, 0, &buffer_dest, 0, size);

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = buffer_dest.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender
                .send(v)
                .expect("Unable to send buffer slice result to async channel.")
        });

        self.device_wgpu.poll(wgpu::Maintain::Wait);

        let result = pollster::block_on(receiver.receive());

        if let Some(Ok(())) = result {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            buffer_dest.unmap();
            result
        } else {
            panic!("Unable to read buffer {:?}", result)
        }
    }

    /// Compile a kernel template if not present in the cache.
    pub fn compile<K: KernelGenerator>(&self) -> Arc<ShaderModule> {
        let mut cache = self.cache.lock().unwrap();
        let template_id = TypeId::of::<K>();

        if let Some(module) = cache.get(&template_id) {
            return module.clone();
        }

        let source = K::generate();

        let module = self
            .device_wgpu
            .create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source.as_ref())),
            });
        let module = Arc::new(module);

        cache.insert(template_id, module.clone());

        module
    }

    /// Execute a kernel using the provided buffers.
    ///
    /// # Notes
    ///
    /// This function isn't safe, buffer can be mutated by the GPU. The users must ensure that a
    /// buffer can be mutated when lauching a compute shaders with write access to a buffer.
    ///
    /// Buffer positions are used as bindings when lauching a compute kernel.
    pub fn execute(&self, work_group: &WorkGroup, kernel: &ShaderModule, buffers: &[&Buffer]) {
        let pipeline = self
            .device_wgpu
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: kernel,
                entry_point: "main",
            });

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

        let mut encoder = self
            .device_wgpu
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute.set_pipeline(&pipeline);
        compute.set_bind_group(0, &bind_group, &[]);

        compute.dispatch_workgroups(work_group.x, work_group.y, work_group.z);
        std::mem::drop(compute);

        self.queue.submit(Some(encoder.finish()));
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
