use alloc::vec::Vec;
use core::marker::PhantomData;
use derive_new::new;

use crate::{
    channel::{ComputeChannel, MutexComputeChannel},
    ComputeServer, Handle,
};

/// The ComputeClient is the entry point to require tasks from the ComputeServer.
/// It should be obtained for a specific device via the Compute struct.
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
    /// Given a handle, returns owned resource as bytes
    pub fn read(&self, handle: &Handle<Server>) -> Vec<u8> {
        self.channel.read(handle)
    }

    /// Given a resource as bytes, stores it and returns the resource handle
    pub fn create(&self, data: Vec<u8>) -> Handle<Server> {
        self.channel.create(data)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them
    pub fn empty(&self, size: usize) -> Handle<Server> {
        self.channel.empty(size)
    }

    /// Executes the `kernel` over the given `handles`.
    pub fn execute(&self, kernel_description: Server::Kernel, handles: &[&Handle<Server>]) {
        self.channel.execute(kernel_description, handles)
    }

    /// Wait for the completion of every task in the server.
    pub fn sync(&self) {
        self.channel.sync()
    }
}
