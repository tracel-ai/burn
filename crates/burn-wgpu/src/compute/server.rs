use super::WgpuStorage;
use alloc::{borrow::Cow, sync::Arc};
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use burn_jit::compute::{JitAutotuneKey, JitKernel, Kernel, WorkGroup};
use burn_tensor::Reader;
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
    tasks_max: usize,
    tasks_count: usize,
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
        tasks_max: usize,
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
            tasks_max,
            tasks_count: 0,
        }
    }

    fn submit(&mut self) {
        let mut new_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        core::mem::swap(&mut new_encoder, &mut self.encoder);

        self.queue.submit(Some(new_encoder.finish()));
        self.tasks_count = 0;

        // Cleanup allocations and deallocations.
        self.memory_management.storage().perform_deallocations();
    }

    fn register_compute(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        bind_group: BindGroup,
        work_group: WorkGroup,
    ) {
        let mut compute = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

        compute.set_pipeline(&pipeline);
        compute.set_bind_group(0, &bind_group, &[]);
        compute.dispatch_workgroups(work_group.x, work_group.y, work_group.z);

        self.tasks_count += 1;
    }

    fn pipeline(&mut self, kernel: Kernel) -> Arc<ComputePipeline> {
        let kernel_id = kernel.id();
        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return pipeline.clone();
        }

        let source = kernel.compile().source;
        let pipeline = self.compile_source(&source);
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
                    compilation_options: Default::default(),
                }),
        )
    }

    fn buffer_reader(&mut self, handle: server::Binding<Self>) -> BufferReader {
        let resource = self.memory_management.get(handle.memory);

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
        self.tasks_count += 1;

        self.submit();

        BufferReader::new(buffer_dest)
    }
}

#[derive(new)]
struct BufferReader {
    buffer: wgpu::Buffer,
}

impl BufferReader {
    #[cfg(target_family = "wasm")]
    async fn read(self, device: alloc::sync::Arc<wgpu::Device>) -> Vec<u8> {
        self.read_async(&device).await
    }

    #[cfg(not(target_family = "wasm"))]
    fn read(self, device: &wgpu::Device) -> Vec<u8> {
        pollster::block_on(self.read_async(device))
    }

    async fn read_async(&self, device: &wgpu::Device) -> Vec<u8> {
        let buffer_slice = self.buffer.slice(..);
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
            self.buffer.unmap();
            result
        } else {
            panic!("Unable to read buffer {:?}", result)
        }
    }
}

impl<MM> ComputeServer for WgpuServer<MM>
where
    MM: MemoryManagement<WgpuStorage>,
{
    type Kernel = Kernel;
    type Storage = WgpuStorage;
    type MemoryManagement = MM;
    type AutotuneKey = JitAutotuneKey;

    fn read(&mut self, binding: server::Binding<Self>) -> Reader<Vec<u8>> {
        #[cfg(target_family = "wasm")]
        {
            let future = self.buffer_reader(binding).read(self.device.clone());
            return Reader::Future(Box::pin(future));
        }

        #[cfg(not(target_family = "wasm"))]
        Reader::Concrete(self.buffer_reader(binding).read(&self.device))
    }

    /// When we create a new handle from existing data, we use custom allocations so that we don't
    /// have to execute the current pending tasks.
    ///
    /// This is important, otherwise the compute passes are going to be too small and we won't be able to
    /// fully utilize the GPU.
    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let handle = server::Handle::new(self.memory_management.reserve(data.len()));
        let binding = handle.clone().binding();

        let buffer_src = Arc::new(self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Buffer Src"),
            contents: data,
            usage: wgpu::BufferUsages::COPY_SRC,
        }));

        let resource = self.memory_management.get(binding.memory);

        self.encoder.copy_buffer_to_buffer(
            &buffer_src,
            0,
            &resource.buffer,
            resource.offset(),
            buffer_src.size(),
        );
        self.tasks_count += 1;

        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle<Self> {
        server::Handle::new(self.memory_management.reserve(size))
    }

    fn execute(&mut self, kernel: Self::Kernel, bindings: Vec<server::Binding<Self>>) {
        let work_group = kernel.launch_settings().workgroup;
        let pipeline = self.pipeline(kernel);
        let group_layout = pipeline.get_bind_group_layout(0);

        let memory_handles = bindings
            .into_iter()
            .map(|binding| self.memory_management.get(binding.memory))
            .collect::<Vec<_>>();

        let entries = memory_handles
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

        self.register_compute(pipeline, bind_group, work_group);

        if self.tasks_count >= self.tasks_max {
            self.submit();
        }
    }

    fn sync(&mut self) {
        self.submit();
        self.device.poll(wgpu::Maintain::Wait);
    }
}
