use super::ComputeChannel;
use crate::server::{ComputeServer, Handle};
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

/// The MutexComputeChannel ensures thread-safety by locking the server
/// on every operation
pub struct MutexComputeChannel<Server> {
    server: Arc<Mutex<Server>>,
}

impl<S> Clone for MutexComputeChannel<S> {
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
        }
    }
}
impl<Server> MutexComputeChannel<Server>
where
    Server: ComputeServer,
{
    /// Create a new mutex compute channel.
    pub fn new(server: Server) -> Self {
        Self {
            server: Arc::new(Mutex::new(server)),
        }
    }
}

impl<Server> ComputeChannel<Server> for MutexComputeChannel<Server>
where
    Server: ComputeServer,
{
    fn read(&self, resource_description: &Handle<Server>) -> Vec<u8> {
        let mut server = self.server.lock();

        server.read(resource_description)
    }

    fn create(&self, resource: Vec<u8>) -> Handle<Server> {
        self.server.lock().create(resource)
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        self.server.lock().empty(size)
    }

    fn execute(&self, kernel_description: Server::Kernel, handles: &[&Handle<Server>]) {
        self.server.lock().execute(kernel_description, handles)
    }

    fn sync(&self) {
        self.server.lock().sync()
    }
}
