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
    BindGroup, Buffer, CommandEncoder, ComputePipeline, DeviceDescriptor, DeviceType,
    ShaderModuleDescriptor,
};

use crate::{
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
    sender: mpsc::SyncSender<Message>,
    _handle: std::thread::JoinHandle<()>,
    pub(crate) device: WgpuDevice,
}

enum Message {
    Compute(ComputeTask),
    ReadBuffer(Arc<Buffer>, mpsc::SyncSender<Vec<u8>>),
    CopyBuffer(Arc<Buffer>, Arc<Buffer>),
}

#[derive(new)]
struct ComputeTask {
    bind_group: BindGroup,
    pipeline: Arc<ComputePipeline>,
    work_group: WorkGroup,
}

struct ContextThread {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    encoder: CommandEncoder,
    tasks: Vec<ComputeTask>,
    receiver: mpsc::Receiver<Message>,
}

impl ContextThread {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        receiver: mpsc::Receiver<Message>,
    ) -> std::thread::JoinHandle<()> {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        let context = Self {
            device,
            queue,
            encoder,
            tasks: Vec::new(),
            receiver,
        };

        std::thread::spawn(|| context.run())
    }

    pub fn run(mut self) {
        loop {
            let message = self.receiver.recv().unwrap();

            match message {
                Message::Compute(task) => self.tasks.push(task),
                Message::CopyBuffer(src, dest) => self.buffer_to_buffer(src, dest),
                Message::ReadBuffer(buffer, sender) => {
                    let bytes = self.read(&buffer);
                    sender.send(bytes).unwrap();
                }
            };
        }
    }

    fn compute_tasks(&mut self) {
        let mut compute = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        for task in self.tasks.iter() {
            compute.set_pipeline(&task.pipeline);
            compute.set_bind_group(0, &task.bind_group, &[]);
            compute.dispatch_workgroups(task.work_group.x, task.work_group.y, task.work_group.z);
        }
        std::mem::drop(compute);
        self.tasks.clear();
    }

    fn execute(&mut self) {
        assert!(
            self.tasks.is_empty(),
            "Tasks should be completed before submiting the current encoder."
        );
        let mut new_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        core::mem::swap(&mut new_encoder, &mut self.encoder);

        self.queue.submit(Some(new_encoder.finish()));
    }

    fn read(&mut self, buffer: &Buffer) -> Vec<u8> {
        let size = buffer.size();
        self.compute_tasks();

        let buffer_dest = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create a command encoder
        self.encoder
            .copy_buffer_to_buffer(&buffer, 0, &buffer_dest, 0, size);

        self.execute();

        let buffer_slice = buffer_dest.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender
                .send(v)
                .expect("Unable to send buffer slice result to async channel.")
        });

        self.device.poll(wgpu::Maintain::Wait);

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

    fn buffer_to_buffer(&mut self, buffer_src: Arc<Buffer>, buffer_dest: Arc<Buffer>) {
        self.encoder
            .copy_buffer_to_buffer(&buffer_src, 0, &buffer_dest, 0, buffer_src.size());
    }
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
        println!("Device {:?}", adapter.get_info());

        let device_wgpu = device.clone();
        let mut limits = wgpu::Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits,
            },
            None,
        ))
        .expect("Unable to request the device with the adapter");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let (sender_message, receiver_message) = std::sync::mpsc::sync_channel(50);

        let thread = ContextThread::new(device.clone(), queue.clone(), receiver_message);

        Self {
            id: IdGenerator::generate(),
            device_wgpu: device,
            device: device_wgpu,
            sender: sender_message,
            _handle: thread,
            cache: Mutex::new(HashMap::new()),
        }
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
            .send(Message::CopyBuffer(Arc::new(buffer_src), buffer.clone()))
            .unwrap();

        buffer
    }

    /// Copy buffer to buffer.
    pub fn buffer_to_buffer(&self, buffer: Arc<Buffer>) -> Arc<Buffer> {
        let buffer_out = self.create_buffer(buffer.size() as usize);

        self.sender
            .send(Message::CopyBuffer(buffer, buffer_out.clone()))
            .unwrap();
        buffer_out
    }

    /// Read a buffer from the GPU and return its content as bytes.
    pub fn buffer_to_data(&self, buffer: Arc<Buffer>) -> Vec<u8> {
        let (sender_message, receiver_message) = std::sync::mpsc::sync_channel(1);

        self.sender
            .send(Message::ReadBuffer(buffer, sender_message))
            .unwrap();

        let mut iter = receiver_message.iter();

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
        let pipeline = Arc::new(pipeline);

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
        let pipeline = Arc::new(pipeline);

        cache.insert(template_id, pipeline.clone());

        pipeline
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

        let task = ComputeTask::new(bind_group, pipeline, work_group);
        self.sender.send(Message::Compute(task)).unwrap();
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
