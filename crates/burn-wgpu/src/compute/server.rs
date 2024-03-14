use super::WgpuStorage;
use crate::compute::webgpu_api::*;
use alloc::{borrow::Cow, sync::Arc};
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use burn_jit::compute::{JitAutotuneKey, JitKernel, Kernel, WorkGroup};
use burn_tensor::Reader;
use hashbrown::HashMap;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer<W: WebGPUApi, MM: MemoryManagement<WgpuStorage<W>>> {
    memory_management: MM,
    device: Arc<W::Device>,
    queue: W::Queue,
    encoder: W::CommandEncoder,
    pipelines: HashMap<String, Arc<W::ComputePipeline>>,
    tasks_max: usize,
    tasks_count: usize,
}

impl<W, MM> WgpuServer<W, MM>
where
    W: WebGPUApi,
    MM: MemoryManagement<WgpuStorage<W>>,
{
    /// Create a new server.
    pub fn new(
        memory_management: MM,
        device: Arc<W::Device>,
        queue: W::Queue,
        tasks_max: usize,
    ) -> Self {
        let encoder = device.create_command_encoder(&CommandEncoderDescriptor {
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
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        core::mem::swap(&mut new_encoder, &mut self.encoder);

        self.queue.submit(Some(new_encoder.finish()));
        self.tasks_count = 0;

        // Cleanup allocations and deallocations.
        self.memory_management.storage().perform_deallocations();
    }

    fn register_compute(
        &mut self,
        pipeline: Arc<W::ComputePipeline>,
        bind_group: W::BindGroup,
        work_group: WorkGroup,
    ) {
        self
            .encoder
            .dispatch_compute_pass(&ComputePassDescriptor {
                    label: None,
                },
                pipeline,
                bind_group,
                work_group,
            );

        self.tasks_count += 1;
    }

    fn pipeline(&mut self, kernel: Kernel) -> Arc<W::ComputePipeline> {
        let kernel_id = kernel.id();
        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return pipeline.clone();
        }

        let source = kernel.compile().source;
        let pipeline = self.compile_source(&source);
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        pipeline
    }

    fn compile_source(&self, source: &str) -> Arc<W::ComputePipeline> {
        let module = self.device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(source)),
        });

        Arc::new(
            self.device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: "main",
                }),
        )
    }

    fn buffer_reader(&mut self, handle: server::Binding<Self>) -> BufferReader<W> {
        let resource = self.memory_management.get(handle.memory);

        let size = resource.size();
        let buffer_dest = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: W::MAP_READ | W::COPY_DST,
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
struct BufferReader<W: WebGPUApi> {
    buffer: W::Buffer,
}

impl<W> BufferReader<W>
where
    W: WebGPUApi,
{
    #[cfg(target_family = "wasm")]
    async fn read(self, device: alloc::sync::Arc<W::Device>) -> Vec<u8> {
        self.buffer.read(&device).await
    }

    #[cfg(not(target_family = "wasm"))]
    fn read(self, device: &W::Device) -> Vec<u8> {
        pollster::block_on(self.buffer.read(device))
    }
}

impl<W, MM> ComputeServer for WgpuServer<W, MM>
where
    W: WebGPUApi,
    MM: MemoryManagement<WgpuStorage<W>>,
{
    type Kernel = Kernel;
    type Storage = WgpuStorage<W>;
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
            usage: W::COPY_SRC,
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
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_binding(),
            })
            .collect::<Vec<_>>();

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
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
        W::device_poll(&self.device);
    }
}
