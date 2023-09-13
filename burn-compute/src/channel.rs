use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{ComputeServer, ServerResource};

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

    fn read(&self, resource_description: &ServerResource<Server>) -> Vec<u8> {
        let mut server = self.server.lock();

        server.read(resource_description)
    }

    fn create(&self, resource: Vec<u8>) -> ServerResource<Server> {
        self.server.lock().create(resource)
    }

    fn empty(&self, size: usize) -> ServerResource<Server> {
        self.server.lock().empty(size)
    }

    fn execute(
        &self,
        kernel_description: Server::KernelDescription,
        resource_descriptions: &[&ServerResource<Server>],
    ) {
        self.server
            .lock()
            .execute(kernel_description, resource_descriptions)
    }

    fn sync(&self) {
        self.server.lock().sync()
    }
}

pub trait ComputeChannel<Server: ComputeServer>: Clone {
    fn new(server: Server) -> Self;
    fn read(&self, resource_description: &ServerResource<Server>) -> Vec<u8>;
    fn create(&self, resource: Vec<u8>) -> ServerResource<Server>;
    fn empty(&self, size: usize) -> ServerResource<Server>;
    fn execute(
        &self,
        kernel_description: Server::KernelDescription,
        resource_descriptions: &[&ServerResource<Server>],
    );
    fn sync(&self);
}
