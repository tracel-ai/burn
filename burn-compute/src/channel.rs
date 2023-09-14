use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{ComputeServer, Handle};

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

impl<Server> ComputeChannel<Server> for MutexComputeChannel<Server>
where
    Server: ComputeServer,
{
    fn new(server: Server) -> Self {
        Self {
            server: Arc::new(Mutex::new(server)),
        }
    }

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

pub trait ComputeChannel<Server: ComputeServer>: Clone {
    fn new(server: Server) -> Self;
    fn read(&self, resource_description: &Handle<Server>) -> Vec<u8>;
    fn create(&self, resource: Vec<u8>) -> Handle<Server>;
    fn empty(&self, size: usize) -> Handle<Server>;
    fn execute(
        &self,
        kernel_description: Server::Kernel,
        resource_descriptions: &[&Handle<Server>],
    );
    fn sync(&self);
}
