use std::{borrow::Cow, sync::Arc};

use super::WgpuStorage;
use crate::{context::WorkGroup, kernel::SourceTemplate};
use burn_compute::{
    memory_management::{MemoryManagement, SimpleMemoryManagement},
    server::{self, ComputeServer},
};
use hashbrown::HashMap;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, CommandEncoder, ComputePipeline, ShaderModuleDescriptor,
};

/// Wgpu compute server.
pub struct WgpuServer<MM = SimpleMemoryManagement<WgpuStorage>> {
    memory_management: MM,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    encoder: CommandEncoder,
    pipelines: HashMap<String, Arc<ComputePipeline>>,
    tasks: Vec<ComputeTask>,
    max_tasks: usize,
}

#[derive(new)]
struct ComputeTask {
    pipeline: Arc<ComputePipeline>,
    bind_group: BindGroup,
    work_group: WorkGroup,
}

pub trait Kernel: 'static + Send {
    /// Source template for the kernel.
    fn source_template(self: Box<Self>) -> SourceTemplate;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    fn workgroup(&self) -> WorkGroup;
}

impl<MM> WgpuServer<MM>
where
    MM: MemoryManagement<WgpuStorage>,
{
    pub fn new(
        memory_management: MM,
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
        max_tasks: usize,
    ) -> Self {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        Self {
            memory_management,
            device,
            queue,
            encoder,
            pipelines: HashMap::new(),
            tasks: Vec::new(),
            max_tasks,
        }
    }

    fn submit(&mut self) {
        assert!(
            self.tasks.is_empty(),
            "Tasks should be completed before submitting the current encoder."
        );
        println!("Submit");
        let mut new_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        core::mem::swap(&mut new_encoder, &mut self.encoder);

        self.queue.submit(Some(new_encoder.finish()));
    }

    fn register_tasks(&mut self) {
        if self.tasks.is_empty() {
            return;
        }

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

    fn pipeline(&mut self, kernel: Box<dyn Kernel>) -> Arc<ComputePipeline> {
        let kernel_id = kernel.id();
        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return pipeline.clone();
        }

        let pipeline = self.compile_source(&kernel.source_template().complete());
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        pipeline
    }

    fn compile_source(&self, source: &str) -> Arc<ComputePipeline> {
        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });

        Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: "main",
                }),
        )
    }
}

impl<MM> ComputeServer for WgpuServer<MM>
where
    MM: MemoryManagement<WgpuStorage>,
{
    type Kernel = Box<dyn Kernel>;
    type Storage = WgpuStorage;
    type MemoryManagement = MM;

    fn read(&mut self, handle: &server::Handle<Self>) -> Vec<u8> {
        // Register previous tasks before reading the buffer so that it is up to date.
        self.register_tasks();

        let resource = self.memory_management.get(handle);

        let size = resource.size();
        let buffer_dest = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.encoder.copy_buffer_to_buffer(
            &resource.buffer,
            resource.offset(),
            &buffer_dest,
            0,
            size,
        );

        self.submit();

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

    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let handle = self.empty(data.len());

        let buffer_src = Arc::new(self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Buffer Src"),
            contents: data,
            usage: wgpu::BufferUsages::COPY_SRC,
        }));

        let resource = self.memory_management.get(&handle);

        self.register_tasks();

        self.encoder.copy_buffer_to_buffer(
            &buffer_src,
            0,
            &resource.buffer,
            resource.offset(),
            buffer_src.size(),
        );

        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle<Self> {
        self.memory_management.reserve(size)
    }

    fn execute(&mut self, kernel: Self::Kernel, handles: &[&server::Handle<Self>]) {
        let work_group = kernel.workgroup();
        let pipeline = self.pipeline(kernel);
        let group_layout = pipeline.get_bind_group_layout(0);

        let handles = handles
            .iter()
            .map(|handle| self.memory_management.get(handle))
            .collect::<Vec<_>>();

        let entries = handles
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_binding(),
            })
            .collect::<Vec<_>>();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries: &entries,
        });

        self.tasks
            .push(ComputeTask::new(pipeline, bind_group, work_group));

        if self.tasks.len() >= self.max_tasks {
            self.register_tasks();
            self.submit();
        }
    }

    fn sync(&mut self) {
        if !self.tasks.is_empty() {
            self.register_tasks();
            self.submit();
        }
    }
}
