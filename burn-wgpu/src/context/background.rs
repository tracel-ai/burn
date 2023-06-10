use super::WorkGroup;
use std::sync::{mpsc, Arc};
use wgpu::{BindGroup, Buffer, CommandEncoder, ComputePipeline};

pub struct ContextBackground {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    encoder: CommandEncoder,
    tasks: Vec<ComputeTask>,
    receiver: mpsc::Receiver<BackgroundTask>,
}

#[derive(new)]
pub struct ComputeTask {
    bind_group: BindGroup,
    pipeline: Arc<ComputePipeline>,
    work_group: WorkGroup,
}

#[derive(new)]
pub struct ReadBufferTask {
    buffer: Arc<Buffer>,
    sender: mpsc::Sender<Vec<u8>>,
}

#[derive(new)]
pub struct CopyBufferTask {
    buffer_src: Arc<Buffer>,
    buffer_dest: Arc<Buffer>,
}

pub enum BackgroundTask {
    Compute(ComputeTask),
    ReadBuffer(ReadBufferTask),
    CopyBuffer(CopyBufferTask),
}

impl Into<BackgroundTask> for ComputeTask {
    fn into(self) -> BackgroundTask {
        BackgroundTask::Compute(self)
    }
}

impl Into<BackgroundTask> for ReadBufferTask {
    fn into(self) -> BackgroundTask {
        BackgroundTask::ReadBuffer(self)
    }
}

impl Into<BackgroundTask> for CopyBufferTask {
    fn into(self) -> BackgroundTask {
        BackgroundTask::CopyBuffer(self)
    }
}

impl ContextBackground {
    pub fn start(
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
        receiver: mpsc::Receiver<BackgroundTask>,
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

    fn run(mut self) {
        loop {
            let message = self.receiver.recv().unwrap();

            match message {
                BackgroundTask::Compute(task) => self.tasks.push(task),
                BackgroundTask::CopyBuffer(task) => {
                    self.buffer_to_buffer(task.buffer_src, task.buffer_dest)
                }
                BackgroundTask::ReadBuffer(task) => {
                    let bytes = self.read(&task.buffer);
                    task.sender.send(bytes).unwrap();
                }
            };

            // Submit the tasks to the GPU when more than 50 tasks are accumulated.
            const MAX_TASKS: usize = 50;

            if self.tasks.len() > MAX_TASKS {
                self.register_tasks();
                self.submit();
            }
        }
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
        self.register_tasks();

        let buffer_dest = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.encoder
            .copy_buffer_to_buffer(&buffer, 0, &buffer_dest, 0, size);

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

    fn buffer_to_buffer(&mut self, buffer_src: Arc<Buffer>, buffer_dest: Arc<Buffer>) {
        self.encoder
            .copy_buffer_to_buffer(&buffer_src, 0, &buffer_dest, 0, buffer_src.size());
    }
}
