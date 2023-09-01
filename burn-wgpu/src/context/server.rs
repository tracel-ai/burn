use super::{client::ContextClient, WorkGroup};
use std::sync::Arc;
use wgpu::{BindGroup, Buffer, CommandEncoder, ComputePipeline};

#[cfg(feature = "async")]
pub use async_server::{AsyncContextServer, ContextTask, CopyBufferTask, ReadBufferTask};

/// Context server allow to run tasks on the GPU.
///
/// # Notes
///
/// There are two implementations of this trait. One is a bit more performant while the other
/// doesn't require std.
///
/// * [Asynchronous server](AsyncContextServer).
/// * [Synchronous server](SyncContextServer).
pub trait ContextServer {
    /// The client where task can be sent to the server for execution.
    type Client: ContextClient;

    /// Start the server and returns its [client](ContextClient).
    fn start(device: Arc<wgpu::Device>, queue: wgpu::Queue) -> Self::Client;
}

/// Context server where each operation is added in a synchronous maner.
#[derive(Debug)]
pub struct SyncContextServer {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    encoder: CommandEncoder,
    tasks: Vec<ComputeTask>,
    max_tasks: usize,
}

/// Basic building block to execute computing tasks on the GPU.
#[derive(new, Debug)]
pub struct ComputeTask {
    bind_group: BindGroup,
    pipeline: Arc<ComputePipeline>,
    work_group: WorkGroup,
}

/// Most of the functions are similar to [server client](IContextClient).
///
/// The main difference comes from the functions are mutable instead of immutable requirering a
/// lock by a sync client or using an async channel with the async client/server.
impl SyncContextServer {
    /// Create a new sync context server.
    pub fn new(device: Arc<wgpu::Device>, queue: wgpu::Queue) -> Self {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // TODO: Support a way to modify this value without std.
        let max_tasks = match std::env::var("BURN_WGPU_MAX_TASKS") {
            Ok(value) => value
                .parse::<usize>()
                .expect("BURN_WGPU_MAX_TASKS should be a positive integer."),
            Err(_) => 16, // 16 tasks by default
        };

        Self {
            device,
            queue,
            encoder,
            tasks: Vec::new(),
            max_tasks,
        }
    }

    pub fn register_compute(&mut self, task: ComputeTask) {
        self.tasks.push(task);

        if self.tasks.len() > self.max_tasks {
            self.register_tasks();
            self.submit();
        }
    }

    pub fn read_buffer(&mut self, buffer: &Buffer) -> Vec<u8> {
        // Register previous tasks before reading the buffer so that it is up to date.
        self.register_tasks();

        let size = buffer.size();
        let buffer_dest = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.encoder
            .copy_buffer_to_buffer(buffer, 0, &buffer_dest, 0, size);

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

    pub fn sync(&mut self) {
        if !self.tasks.is_empty() {
            self.register_tasks();
            self.submit();
        }

        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn buffer_to_buffer(&mut self, buffer_src: Arc<Buffer>, buffer_dest: Arc<Buffer>) {
        self.encoder
            .copy_buffer_to_buffer(&buffer_src, 0, &buffer_dest, 0, buffer_src.size());
    }

    fn register_tasks(&mut self) {
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
    }
}

#[cfg(feature = "async")]
mod async_server {
    use crate::context::client::AsyncContextClient;

    use super::{ComputeTask, ContextServer, SyncContextServer};
    use std::sync::{mpsc, Arc};
    use wgpu::Buffer;

    #[derive(new)]
    pub struct ReadBufferTask {
        buffer: Arc<Buffer>,
        sender: mpsc::Sender<Vec<u8>>,
    }

    #[derive(new)]
    pub struct CopyBufferTask {
        pub(crate) buffer_src: Arc<Buffer>,
        pub(crate) buffer_dest: Arc<Buffer>,
    }

    pub enum ContextTask {
        Compute(ComputeTask),
        ReadBuffer(ReadBufferTask),
        CopyBuffer(CopyBufferTask),
        Sync(mpsc::Sender<()>),
    }

    impl From<ComputeTask> for ContextTask {
        fn from(val: ComputeTask) -> Self {
            ContextTask::Compute(val)
        }
    }

    impl From<ReadBufferTask> for ContextTask {
        fn from(val: ReadBufferTask) -> Self {
            ContextTask::ReadBuffer(val)
        }
    }

    impl From<CopyBufferTask> for ContextTask {
        fn from(val: CopyBufferTask) -> Self {
            ContextTask::CopyBuffer(val)
        }
    }

    /// Asynchronous context server where [tasks](ContextTask) are sent using a channel.
    ///
    /// # Notes
    ///
    /// This is pretty useful to avoid blocking the main thread when registering and
    /// executing [compute tasks](ComputeTask).
    pub struct AsyncContextServer {
        server: SyncContextServer,
        receiver: mpsc::Receiver<ContextTask>,
    }

    impl AsyncContextServer {
        fn run(mut self) {
            loop {
                let task = self.receiver.recv().unwrap();
                match task {
                    ContextTask::Compute(task) => self.server.register_compute(task),
                    ContextTask::CopyBuffer(task) => self
                        .server
                        .buffer_to_buffer(task.buffer_src, task.buffer_dest),
                    ContextTask::ReadBuffer(task) => {
                        let bytes = self.server.read_buffer(&task.buffer);
                        task.sender.send(bytes).unwrap();
                    }
                    ContextTask::Sync(callback) => {
                        self.server.sync();
                        callback.send(()).unwrap();
                    }
                };
            }
        }
    }
    impl ContextServer for AsyncContextServer {
        type Client = AsyncContextClient;

        fn start(device: Arc<wgpu::Device>, queue: wgpu::Queue) -> Self::Client {
            let (sender, receiver) = std::sync::mpsc::sync_channel(50);
            let server = SyncContextServer::new(device, queue);
            let context = Self { server, receiver };

            let handle = std::thread::spawn(|| context.run());

            AsyncContextClient::new(sender, handle)
        }
    }
}

#[cfg(not(feature = "async"))]
mod sync_server {
    use super::{ContextServer, SyncContextServer};
    use crate::context::client::SyncContextClient;
    use std::sync::Arc;

    impl ContextServer for SyncContextServer {
        type Client = SyncContextClient;

        fn start(device: Arc<wgpu::Device>, queue: wgpu::Queue) -> Self::Client {
            let server = Self::new(device, queue);

            SyncContextClient::new(server)
        }
    }
}
