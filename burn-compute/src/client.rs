use crate::{
    channel::ComputeChannel,
    server::{ComputeServer, Handle},
    tune::{AutotuneOperationSet, Tuner},
};
use alloc::vec::Vec;
use alloc::{boxed::Box, sync::Arc};
use burn_common::reader::Reader;
use core::marker::PhantomData;
use spin::Mutex;

/// The ComputeClient is the entry point to require tasks from the ComputeServer.
/// It should be obtained for a specific device via the Compute struct.
#[derive(Debug)]
pub struct ComputeClient<Server, Channel> {
    channel: Channel,
    tuner: Arc<Mutex<Tuner<Server, Channel>>>,
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
            tuner: self.tuner.clone(),
            _server: PhantomData,
        }
    }
}

impl<Server, Channel> ComputeClient<Server, Channel>
where
    Server: ComputeServer,
    Channel: ComputeChannel<Server>,
{
    /// Create a new client.
    pub fn new(channel: Channel, tuner: Arc<Mutex<Tuner<Server, Channel>>>) -> Self {
        Self {
            channel,
            tuner,
            _server: PhantomData,
        }
    }

    /// Given a handle, returns owned resource as bytes.
    pub fn read(&self, handle: &Handle<Server>) -> Reader<Vec<u8>> {
        self.channel.read(handle)
    }

    /// Given a resource, stores it and returns the resource handle.
    pub fn create(&self, data: &[u8]) -> Handle<Server> {
        self.channel.create(data)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle<Server> {
        self.channel.empty(size)
    }

    /// Executes the `kernel` over the given `handles`.
    pub fn execute(&self, kernel: Server::Kernel, handles: &[&Handle<Server>]) {
        self.channel.execute(kernel, handles)
    }

    /// Wait for the completion of every task in the server.
    pub fn sync(&self) {
        self.channel.sync()
    }

    /// Executes the fastest kernel in the autotune operation, using (cached) runtime benchmarks
    pub fn execute_autotune(&self, autotune_operation_set: Box<dyn AutotuneOperationSet>) {
        self.tuner
            .lock()
            .execute_autotune(autotune_operation_set, self);
    }
}
