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
    fn read(&self, handle: &Handle<Server>) -> Vec<u8> {
        self.server.lock().read(handle)
    }

    fn create(&self, data: &[u8]) -> Handle<Server> {
        self.server.lock().create(data)
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        self.server.lock().empty(size)
    }

    fn execute(&self, kernel: Server::Kernel, handles: &[&Handle<Server>]) {
        self.server.lock().execute(kernel, handles)
    }

    fn sync(&self) {
        self.server.lock().sync()
    }
}
