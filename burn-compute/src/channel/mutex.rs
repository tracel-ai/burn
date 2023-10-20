use super::ComputeChannel;
use crate::server::{ComputeServer, Handle};
use crate::tune::AutotuneServer;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use burn_common::reader::Reader;
use spin::Mutex;

/// The MutexComputeChannel ensures thread-safety by locking the server
/// on every operation
#[derive(Debug)]
pub struct MutexComputeChannel<Server> {
    autotune_server: Arc<Mutex<AutotuneServer<Server>>>,
}

impl<S> Clone for MutexComputeChannel<S> {
    fn clone(&self) -> Self {
        Self {
            autotune_server: self.autotune_server.clone(),
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
            autotune_server: Arc::new(Mutex::new(AutotuneServer::new(server))),
        }
    }
}

impl<Server> ComputeChannel<Server> for MutexComputeChannel<Server>
where
    Server: ComputeServer,
{
    fn read(&self, handle: &Handle<Server>) -> Reader<Vec<u8>> {
        self.autotune_server.lock().server.read(handle)
    }

    fn create(&self, data: &[u8]) -> Handle<Server> {
        self.autotune_server.lock().server.create(data)
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        self.autotune_server.lock().server.empty(size)
    }

    fn execute(&self, kernel: Server::Kernel, handles: &[&Handle<Server>]) {
        self.autotune_server.lock().server.execute(kernel, handles)
    }

    fn sync(&self) {
        self.autotune_server.lock().server.sync()
    }

    fn execute_autotune(
        &self,
        autotune_kernel: Box<dyn crate::tune::AutotuneOperation<Server>>,
        handles: &[&Handle<Server>],
    ) {
        self.autotune_server
            .lock()
            .execute_autotune(autotune_kernel, handles);
    }
}
