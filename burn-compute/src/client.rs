use alloc::vec::Vec;
use core::marker::PhantomData;
use derive_new::new;

use crate::{
    channel::{ComputeChannel, MutexComputeChannel},
    ComputeServer, Handle,
};

#[derive(new)]
pub struct ComputeClient<Server, Channel = MutexComputeChannel<Server>> {
    channel: Channel,
    _server: PhantomData<Server>,
}

impl<S, C> Clone for ComputeClient<S, C>
where
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    fn clone(&self) -> Self {
        Self {
            channel: self.channel.clone(),
            _server: PhantomData,
        }
    }
}

impl<Server> ComputeClient<Server>
where
    Server: ComputeServer,
{
    pub fn read(&self, resource_description: &Handle<Server>) -> Vec<u8> {
        self.channel.read(resource_description)
    }

    pub fn create(&self, resource: Vec<u8>) -> Handle<Server> {
        self.channel.create(resource)
    }

    pub fn empty(&self, size: usize) -> Handle<Server> {
        self.channel.empty(size)
    }

    pub fn execute(
        &self,
        kernel_description: Server::KernelDescription,
        resource_descriptions: &[&Handle<Server>],
    ) {
        self.channel
            .execute(kernel_description, resource_descriptions)
    }

    pub fn sync(&self) {
        self.channel.sync()
    }
}
