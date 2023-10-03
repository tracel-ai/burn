use super::{WgpuStorage, WorkGroup};
use crate::kernel::SourceTemplate;
use alloc::{borrow::Cow, sync::Arc};
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use hashbrown::HashMap;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, CommandEncoder, ComputePipeline, ShaderModuleDescriptor,
};

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer<MM: MemoryManagement<WgpuStorage>> {
    memory_management: MM,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    encoder: CommandEncoder,
    pipelines: HashMap<String, Arc<ComputePipeline>>,
    tasks: Vec<ComputeTask>,
    max_tasks: usize,
    manual_available: HashMap<usize, Vec<server::Handle<Self>>>,
    manual_taken: Vec<(usize, server::Handle<Self>)>,
}

#[derive(new, Debug)]
struct ComputeTask {
    pipeline: Arc<ComputePipeline>,
    bind_group: BindGroup,
    work_group: WorkGroup,
}

/// Kernel trait with the [source](SourceTemplate) that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [workgroup](WorkGroup).
pub trait Kernel: 'static + Send {
    /// Source template for the kernel.
    fn source(self: Box<Self>) -> SourceTemplate;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Launch information.
    fn workgroup(&self) -> WorkGroup;
}

impl<MM> WgpuServer<MM>
where
    MM: MemoryManagement<WgpuStorage>,
{
    /// Create a new server.
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
            manual_available: HashMap::new(),
            manual_taken: Vec::new(),
        }
    }

    fn submit(&mut self) {
        assert!(
            self.tasks.is_empty(),
            "Tasks should be completed before submitting the current encoder."
        );
        let mut new_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        core::mem::swap(&mut new_encoder, &mut self.encoder);

        self.queue.submit(Some(new_encoder.finish()));

        // Cleanup allocations and deallocations.
        self.free_manual_allocations();
        self.memory_management.storage().perform_deallocations();
    }

    fn free_manual_allocations(&mut self) {
        let mut manual_taken_tmp = Vec::new();
        core::mem::swap(&mut manual_taken_tmp, &mut self.manual_taken);

        for (size, handle) in manual_taken_tmp.drain(..) {
            if handle.can_mut() {
                self.register_manual(size, handle);
            } else {
                self.manual_taken.push((size, handle));
            }
        }
    }

    // Finds a free, manually-added handle of specified size, or creates it if none is found
    fn manual_reserve(&mut self, size: usize) -> server::Handle<Self> {
        let handle = self
            .manual_available
            .get_mut(&size)
            .and_then(|h| h.pop())
            .unwrap_or_else(|| {
                let memory = self.memory_management.alloc(size);
                server::Handle::new(memory)
            });

        self.manual_taken.push((size, handle.clone()));

        handle
    }

    // Manually adds a handle of given size
    fn register_manual(&mut self, size: usize, handle: server::Handle<Self>) {
        if let Some(handles) = self.manual_available.get_mut(&size) {
            handles.push(handle);
        } else {
            self.manual_available.insert(size, [handle].into());
        }
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

        let pipeline = self.compile_source(&kernel.source().complete());
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

        let resource = self.memory_management.get(&handle.memory);

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

    /// When we create a new handle from existing data, we use custom allocations so that we don't
    /// have to execute the current pending tasks.
    ///
    /// This is important, otherwise the compute passes are going to be too small and we won't be able to
    /// fully utilize the GPU.
    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let handle = self.manual_reserve(data.len());

        let buffer_src = Arc::new(self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Buffer Src"),
            contents: data,
            usage: wgpu::BufferUsages::COPY_SRC,
        }));

        let resource = self.memory_management.get(&handle.memory);

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
        server::Handle::new(self.memory_management.reserve(size))
    }

    fn execute(&mut self, kernel: Self::Kernel, handles: &[&server::Handle<Self>]) {
        let work_group = kernel.workgroup();
        let pipeline = self.pipeline(kernel);
        let group_layout = pipeline.get_bind_group_layout(0);

        let handles = handles
            .iter()
            .map(|handle| self.memory_management.get(&handle.memory))
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

        self.device.poll(wgpu::Maintain::Wait);
    }
}
