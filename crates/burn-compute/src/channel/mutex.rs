use super::ComputeChannel;
use crate::server::{Binding, ComputeServer, Handle};
use crate::storage::ComputeStorage;
use alloc::sync::Arc;
use alloc::vec::Vec;
use burn_common::reader::Reader;
use burn_common::sync_type::SyncType;
use spin::Mutex;

/// The MutexComputeChannel ensures thread-safety by locking the server
/// on every operation
#[derive(Debug)]
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
    fn read(&self, handle: Binding<Server>) -> Reader<Vec<u8>> {
        self.server.lock().read(handle)
    }

    fn get_resource(
        &self,
        binding: Binding<Server>,
    ) -> <Server::Storage as ComputeStorage>::Resource {
        self.server.lock().get_resource(binding)
    }

    fn create(&self, data: &[u8]) -> Handle<Server> {
        self.server.lock().create(data)
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        self.server.lock().empty(size)
    }

    fn execute(&self, kernel: Server::Kernel, handles: Vec<Binding<Server>>) {
        self.server.lock().execute(kernel, handles)
    }

    fn sync(&self, sync_type: SyncType) {
        self.server.lock().sync(sync_type)
    }
}
