use std::num::NonZeroU64;

use super::WgpuStorage;
use alloc::{borrow::Cow, sync::Arc};
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use burn_cube::prelude::*;
use burn_jit::JitAutotuneKey;
use burn_tensor::{backend::SyncType, Reader};
use hashbrown::HashMap;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt, StagingBelt},
    BindGroup, CommandEncoder, ComputePipeline, ShaderModuleDescriptor,
};

// Allocations with existing data smaller than this can use a staging belt
// which speeds up the allocation. A higher number here will catch more
// allocations, but can also increase memory usage.
const SMALL_ALLOC_SIZE: usize = 512;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer<MM: MemoryManagement<WgpuStorage>> {
    memory_management: MM,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    encoder: CommandEncoder,
    staging_belt: StagingBelt,
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
        queue: Arc<wgpu::Queue>,
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
            staging_belt: StagingBelt::new(SMALL_ALLOC_SIZE as u64),
            pipelines: HashMap::new(),
            tasks_max,
            tasks_count: 0,
        }
    }

    fn register_compute(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        bind_group: BindGroup,
        work_group: CubeCount,
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

    fn pipeline(&mut self, kernel: Box<dyn CubeTask>) -> Arc<ComputePipeline> {
        let kernel_id = kernel.id();

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return pipeline.clone();
        }

        let compile = kernel.compile();
        let pipeline = self.compile_source(&compile.source);

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

        self.sync(SyncType::Flush);

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
    type Kernel = Box<dyn CubeTask>;
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

    fn get_resource(
        &mut self,
        binding: server::Binding<Self>,
    ) -> <Self::Storage as burn_compute::storage::ComputeStorage>::Resource {
        self.memory_management.get(binding.memory)
    }

    /// When we create a new handle from existing data, we use custom allocations so that we don't
    /// have to execute the current pending tasks.
    ///
    /// This is important, otherwise the compute passes are going to be too small and we won't be able to
    /// fully utilize the GPU.
    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let handle = server::Handle::new(self.memory_management.reserve(data.len(), || {
            flush_tasks(
                &mut self.encoder,
                &self.queue,
                &self.device,
                &mut self.tasks_count,
                &mut self.staging_belt,
            );
            self.device.poll(wgpu::Maintain::Wait);
        }));

        let non_zero_len = NonZeroU64::new(data.len() as u64);

        // If there's nothing to copy, don't need to do any work here.
        if let Some(len) = non_zero_len {
            let binding = handle.clone().binding();
            let resource = self.memory_management.get(binding.memory);

            if data.len() < SMALL_ALLOC_SIZE {
                // Use a staging belt if the allocation is small enough. This is faster than allocating a new buffer.
                // Ideally, we could use queue.write_buffer_with(), which seems to be the recommended method for performance,
                // but that doesn't seem to work, as we might re-use a buffer multiple times, and need to schedule this
                // precisely in the encoder.
                let mut write_buf = self.staging_belt.write_buffer(
                    &mut self.encoder,
                    &resource.buffer,
                    resource.offset(),
                    len,
                    &self.device,
                );
                write_buf.copy_from_slice(data);
            } else {
                let buffer_src = Arc::new(self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Buffer Src"),
                    contents: data,
                    usage: wgpu::BufferUsages::COPY_SRC,
                }));
                self.encoder.copy_buffer_to_buffer(
                    &buffer_src,
                    0,
                    &resource.buffer,
                    resource.offset(),
                    buffer_src.size(),
                );
            }
            self.tasks_count += 1;
        }

        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle<Self> {
        server::Handle::new(self.memory_management.reserve(size, || {
            flush_tasks(
                &mut self.encoder,
                &self.queue,
                &self.device,
                &mut self.tasks_count,
                &mut self.staging_belt,
            );
            self.device.poll(wgpu::Maintain::Wait);
        }))
    }

    fn execute(&mut self, kernel: Self::Kernel, bindings: Vec<server::Binding<Self>>) {
        let work_group = kernel.launch_settings().cube_count;
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
            self.sync(SyncType::Flush);
        }
    }

    fn sync(&mut self, sync_type: SyncType) {
        flush_tasks(
            &mut self.encoder,
            &self.queue,
            &self.device,
            &mut self.tasks_count,
            &mut self.staging_belt,
        );

        // Cleanup allocations and deallocations.
        self.memory_management.storage().perform_deallocations();

        if sync_type == SyncType::Wait {
            self.device.poll(wgpu::Maintain::Wait);
        }
    }
}

/// Flush tasks using the [command encoder](CommandEncoder).
///
/// This implementation is decoupled from both the [server](WgpuServer) and [memory management](MemoryManagement).
/// This decoupling allows for safe usage within sync callbacks when allocating memory buffers.
fn flush_tasks(
    encoder: &mut CommandEncoder,
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    tasks_count: &mut usize,
    staging_belt: &mut StagingBelt,
) {
    staging_belt.finish();

    let mut new_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    core::mem::swap(&mut new_encoder, encoder);

    queue.submit(Some(new_encoder.finish()));
    *tasks_count = 0;
    staging_belt.recall();
}
