use super::WgpuStorage;
use crate::compute::webgpu_api::*;
use alloc::{borrow::Cow, sync::Arc};
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use burn_jit::compute::{JitAutotuneKey, Kernel, WorkGroup};
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
    tasks: Vec<ComputeTask<W::BindGroup, W::ComputePipeline>>,
    max_tasks: usize,
    manual_available: HashMap<usize, Vec<server::Handle<Self>>>,
    manual_taken: Vec<(usize, server::Handle<Self>)>,
}

#[derive(new, Debug)]
pub struct ComputeTask<BindGroup, ComputePipeline> {
    pub pipeline: Arc<ComputePipeline>,
    pub bind_group: BindGroup,
    pub work_group: WorkGroup,
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
        max_tasks: usize,
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
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
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

        self.encoder
            .dispatch_compute_pass(&ComputePassDescriptor { label: None }, &self.tasks);

        self.tasks.clear();
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

    fn buffer_reader(&mut self, handle: &server::Handle<Self>) -> BufferReader<W> {
        // Register previous tasks before reading the buffer so that it is up to date.
        self.register_tasks();

        let resource = self.memory_management.get(&handle.memory);

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

    fn read(&mut self, handle: &server::Handle<Self>) -> Reader<Vec<u8>> {
        #[cfg(target_family = "wasm")]
        {
            let future = self.buffer_reader(handle).read(self.device.clone());
            return Reader::Future(Box::pin(future));
        }

        #[cfg(not(target_family = "wasm"))]
        Reader::Concrete(self.buffer_reader(handle).read(&self.device))
    }

    /// When we create a new handle from existing data, we use custom allocations so that we don't
    /// have to execute the current pending tasks.
    ///
    /// This is important, otherwise the compute passes are going to be too small and we won't be able to
    /// fully utilize the GPU.
    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let handle = self.manual_reserve(data.len());

        let resource = self.memory_management.get(&handle.memory);

        self.queue.write_buffer(resource.buffer.as_ref(), 0, data);

        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle<Self> {
        server::Handle::new(self.memory_management.reserve(size))
    }

    fn execute(&mut self, kernel: Self::Kernel, handles: &[&server::Handle<Self>]) {
        let work_group = kernel.launch_settings().workgroup;
        let pipeline = self.pipeline(kernel);
        let group_layout = pipeline.get_bind_group_layout(0);

        let handles = handles
            .iter()
            .map(|handle| self.memory_management.get(&handle.memory))
            .collect::<Vec<_>>();

        let entries = handles
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

        W::device_poll(&self.device);
    }
}
