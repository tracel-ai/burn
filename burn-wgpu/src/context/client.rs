use super::WorkGroup;
use std::sync::Arc;
use wgpu::{BindGroup, Buffer, ComputePipeline};

#[cfg(feature = "async")]
pub use async_client::AsyncContextClient;
#[cfg(not(feature = "async"))]
pub use sync_client::SyncContextClient;

/// Context client allows to speak with a server to execute tasks on the GPU.
pub trait ContextClient {
    /// Copy the source buffer content into the destination buffer.
    ///
    /// # Notes
    ///
    /// Make sure the source buffer isn't used afterward, since a race condition may happen.
    ///
    /// If the source buffer is still used afterward, use [tensor copy](crate::tensor::WgpuTensor::copy)
    /// instead. This method is still useful to load data from the CPU into a new buffer.
    fn copy_buffer(
        &self,
        buffer_src: Arc<Buffer>,
        buffer_dest: Arc<Buffer>,
        wait_for_registered: bool,
    ) -> Arc<Buffer>;
    /// Read a [buffer](Buffer).
    ///
    /// # Notes
    ///
    /// All pending compute tasks will be executed.
    fn read_buffer(&self, buffer: Arc<Buffer>) -> Vec<u8>;
    /// Register a new computing task.
    fn register_compute(
        &self,
        bind_group: BindGroup,
        pipeline: Arc<ComputePipeline>,
        work_group: WorkGroup,
    );
    /// Wait for all computation to be done.
    ///
    /// Useful for benchmarks.
    fn sync(&self);
}

#[cfg(feature = "async")]
mod async_client {
    use super::ContextClient;
    use crate::context::{
        server::{ComputeTask, ContextTask, CopyBufferTask, ReadBufferTask},
        WorkGroup,
    };
    use std::sync::{mpsc, Arc};
    use wgpu::{BindGroup, Buffer, ComputePipeline};

    /// Client returned by
    #[derive(new, Debug)]
    pub struct AsyncContextClient {
        sender: mpsc::SyncSender<ContextTask>,
        _server_handle: std::thread::JoinHandle<()>,
    }

    impl ContextClient for AsyncContextClient {
        fn sync(&self) {
            let (sender, receiver) = std::sync::mpsc::channel();

            self.sender.send(ContextTask::Sync(sender)).unwrap();

            if receiver.iter().next().is_some() {
                log::debug!("Sync completed");
            } else {
                panic!("Unable sync")
            }
        }

        fn copy_buffer(
            &self,
            buffer_src: Arc<Buffer>,
            buffer_dest: Arc<Buffer>,
            wait_for_registered: bool,
        ) -> Arc<Buffer> {
            if wait_for_registered {
                assert_eq!(Arc::strong_count(&buffer_dest), 1, "You can't wait for the buffer to be registered when multiple references already exist.");
            }

            self.sender
                .send(CopyBufferTask::new(buffer_src, buffer_dest.clone()).into())
                .unwrap();

            if !wait_for_registered {
                return buffer_dest;
            }

            // Wait for the buffer to be correctly registered so that inplace operations can be
            // prioritize.
            //
            // Note that this is unsafe and a channel could have been used to wait for completion.
            // The loop is there for performance reason.
            //
            // TODO: Use a performant one time channel here as callback instead.
            loop {
                std::thread::sleep(std::time::Duration::from_micros(1));

                if Arc::strong_count(&buffer_dest) == 1 {
                    return buffer_dest;
                }
            }
        }

        fn read_buffer(&self, buffer: Arc<Buffer>) -> Vec<u8> {
            let (sender, receiver) = std::sync::mpsc::channel();

            self.sender
                .send(ReadBufferTask::new(buffer, sender).into())
                .unwrap();

            let mut iter = receiver.iter();
            if let Some(data) = iter.next() {
                data
            } else {
                panic!("Unable to read buffer")
            }
        }
        fn register_compute(
            &self,
            bind_group: BindGroup,
            pipeline: Arc<ComputePipeline>,
            work_group: WorkGroup,
        ) {
            self.sender
                .send(ComputeTask::new(bind_group, pipeline, work_group).into())
                .unwrap();
        }
    }
}

#[cfg(not(feature = "async"))]
mod sync_client {
    use super::ContextClient;
    use crate::context::{
        server::{ComputeTask, SyncContextServer},
        WorkGroup,
    };
    use std::sync::Arc;
    use wgpu::{BindGroup, Buffer, ComputePipeline};

    #[derive(Debug)]
    pub struct SyncContextClient {
        server: spin::Mutex<SyncContextServer>,
    }

    impl SyncContextClient {
        pub fn new(server: SyncContextServer) -> Self {
            Self {
                server: spin::Mutex::new(server),
            }
        }
    }

    impl ContextClient for SyncContextClient {
        fn sync(&self) {
            let mut server = self.server.lock();
            server.sync();
        }
        fn copy_buffer(
            &self,
            buffer_src: Arc<Buffer>,
            buffer_dest: Arc<Buffer>,
            _wait_for_registered: bool, // Ignored when sync
        ) -> Arc<Buffer> {
            let mut server = self.server.lock();
            server.buffer_to_buffer(buffer_src, buffer_dest.clone());

            buffer_dest
        }
        fn read_buffer(&self, buffer: Arc<Buffer>) -> Vec<u8> {
            let mut server = self.server.lock();
            server.read_buffer(&buffer)
        }

        fn register_compute(
            &self,
            bind_group: BindGroup,
            pipeline: Arc<ComputePipeline>,
            work_group: WorkGroup,
        ) {
            let mut server = self.server.lock();
            server.register_compute(ComputeTask::new(bind_group, pipeline, work_group));
        }
    }
}
