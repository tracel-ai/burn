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
    pub fn read(&self, handle: &Handle<Server>) -> Vec<u8> {
        self.channel.read(handle)
    }

    pub fn create(&self, data: Vec<u8>) -> Handle<Server> {
        self.channel.create(data)
    }

    pub fn empty(&self, size: usize) -> Handle<Server> {
        self.channel.empty(size)
    }

    pub fn execute(&self, kernel_description: Server::Kernel, handles: &[&Handle<Server>]) {
        self.channel.execute(kernel_description, handles)
    }

    pub fn sync(&self) {
        self.channel.sync()
    }
}
