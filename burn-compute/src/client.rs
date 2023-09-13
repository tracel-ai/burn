use alloc::vec::Vec;
use core::marker::PhantomData;
use derive_new::new;

use crate::{
    channel::{ComputeChannel, MutexComputeChannel},
    ComputeServer,
};

#[derive(new)]
pub struct ComputeClient<Server, Channel = MutexComputeChannel<Server>> {
    channel: Channel,
    _server: PhantomData<Server>,
}

impl<Server> ComputeClient<Server>
where
    Server: ComputeServer,
{
    pub fn read(&self, resource_description: &Server::ResourceDescription) -> Vec<u8> {
        self.channel.read(resource_description)
    }

    pub fn create(&self, resource: Vec<u8>) -> Server::ResourceDescription {
        self.channel.create(resource)
    }

    pub fn empty(&self, size: usize) -> Server::ResourceDescription {
        self.channel.empty(size)
    }

    pub fn execute(
        &self,
        kernel_description: Server::KernelDescription,
        resource_descriptions: Vec<&Server::ResourceDescription>,
    ) {
        self.channel
            .execute(kernel_description, resource_descriptions)
    }

    pub fn sync(&self) {
        self.channel.sync()
    }
}
