#[cfg(feature = "async")]
pub use async_client::AsyncContextClient;
#[cfg(feature = "async")]
pub type ContextClient = AsyncContextClient;

#[cfg(not(feature = "async"))]
pub use sync_client::SyncContextClient;
#[cfg(not(feature = "async"))]
pub type ContextClient = SyncContextClient;

#[cfg(feature = "async")]
mod async_client {
    use crate::context::{
        server::{ComputeTask, ContextTask, CopyBufferTask, ReadBufferTask},
        WorkGroup,
    };
    use std::sync::{mpsc, Arc};
    use wgpu::{BindGroup, Buffer, ComputePipeline};

    #[derive(new, Debug)]
    pub struct AsyncContextClient {
        sender: mpsc::SyncSender<ContextTask>,
        _server_handle: std::thread::JoinHandle<()>,
    }

    impl AsyncContextClient {
        pub fn copy_buffer(
            &self,
            buffer_src: Arc<Buffer>,
            buffer_dest: Arc<Buffer>,
            wait_for_registered: bool,
        ) -> Arc<Buffer> {
            self.sender
                .send(CopyBufferTask::new(buffer_src, buffer_dest.clone()).into())
                .unwrap();

            if !wait_for_registered {
                return buffer_dest;
            }

            // Wait for the buffer to be correctly registered so that inplace operations can be
            // prioritize.
            loop {
                std::thread::sleep(std::time::Duration::from_micros(1));

                if Arc::strong_count(&buffer_dest) == 1 {
                    return buffer_dest;
                }
            }
        }

        pub fn compute(
            &self,
            bind_group: BindGroup,
            pipeline: Arc<ComputePipeline>,
            work_group: WorkGroup,
        ) {
            self.sender
                .send(ComputeTask::new(bind_group, pipeline, work_group).into())
                .unwrap();
        }

        pub fn read(&self, buffer: Arc<Buffer>) -> Vec<u8> {
            let (sender, receiver) = std::sync::mpsc::channel();

            self.sender
                .send(ReadBufferTask::new(buffer, sender).into())
                .unwrap();

            let mut iter = receiver.iter();
            if let Some(data) = iter.next() {
                return data;
            } else {
                panic!("Unable to read buffer")
            }
        }
    }
}

#[cfg(not(feature = "async"))]
mod sync_client {
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

        pub fn copy_buffer(
            &self,
            buffer_src: Arc<Buffer>,
            buffer_dest: Arc<Buffer>,
            _wait_for_registered: bool, // Ignored when sync
        ) -> Arc<Buffer> {
            let mut server = self.server.lock();
            server.buffer_to_buffer(buffer_src, buffer_dest.clone());

            buffer_dest
        }

        pub fn compute(
            &self,
            bind_group: BindGroup,
            pipeline: Arc<ComputePipeline>,
            work_group: WorkGroup,
        ) {
            let mut server = self.server.lock();
            server.register_compute(ComputeTask::new(bind_group, pipeline, work_group));
        }

        pub fn read(&self, buffer: Arc<Buffer>) -> Vec<u8> {
            let mut server = self.server.lock();
            server.read(&buffer)
        }
    }
}
