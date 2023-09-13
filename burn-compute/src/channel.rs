use alloc::vec::Vec;
use spin::Mutex;

use crate::ComputeServer;

pub struct MutexComputeChannel<Server> {
    server: Mutex<Server>,
}

impl<Server> ComputeChannel<Server> for MutexComputeChannel<Server>
where
    Server: ComputeServer,
{
    fn new(server: Server) -> Self {
        Self {
            server: Mutex::new(server),
        }
    }

    fn read(&self, resource_description: &Server::ResourceDescription) -> Vec<u8> {
        let mut server = self.server.lock();

        server.read(resource_description)
    }

    fn create(&self, resource: Vec<u8>) -> Server::ResourceDescription {
        self.server.lock().create(resource)
    }

    fn empty(&self, size: usize) -> Server::ResourceDescription {
        self.server.lock().empty(size)
    }

    fn execute(
        &self,
        kernel_description: Server::KernelDescription,
        resource_descriptions: Vec<&Server::ResourceDescription>,
    ) {
        self.server
            .lock()
            .execute(kernel_description, resource_descriptions)
    }

    fn sync(&self) {
        self.server.lock().sync()
    }
}

pub trait ComputeChannel<Server: ComputeServer> {
    fn new(server: Server) -> Self;
    fn read(&self, resource_description: &Server::ResourceDescription) -> Vec<u8>;
    fn create(&self, resource: Vec<u8>) -> Server::ResourceDescription;
    fn empty(&self, size: usize) -> Server::ResourceDescription;
    fn execute(
        &self,
        kernel_description: Server::KernelDescription,
        resource_descriptions: Vec<&Server::ResourceDescription>,
    );
    fn sync(&self);
}
