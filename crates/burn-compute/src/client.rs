use crate::{
    channel::ComputeChannel,
    server::{Binding, ComputeServer, Handle},
    tune::{AutotuneOperationSet, Tuner},
};
use alloc::vec::Vec;
use alloc::{boxed::Box, sync::Arc};
use burn_common::reader::Reader;
use burn_common::stub::RwLock;

/// The ComputeClient is the entry point to require tasks from the ComputeServer.
/// It should be obtained for a specific device via the Compute struct.
#[derive(Debug)]
pub struct ComputeClient<Server: ComputeServer, Channel> {
    channel: Channel,
    tuner: Arc<RwLock<Tuner<Server, Channel>>>,
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
        }
    }
}

impl<Server, Channel> ComputeClient<Server, Channel>
where
    Server: ComputeServer,
    Channel: ComputeChannel<Server>,
{
    /// Create a new client.
    pub fn new(channel: Channel, tuner: Arc<RwLock<Tuner<Server, Channel>>>) -> Self {
        Self { channel, tuner }
    }

    /// Given a binding, returns owned resource as bytes.
    pub fn read(&self, binding: Binding<Server>) -> Reader<Vec<u8>> {
        self.channel.read(binding)
    }

    /// Given a resource, stores it and returns the resource handle.
    pub fn create(&self, data: &[u8]) -> Handle<Server> {
        self.channel.create(data)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle<Server> {
        self.channel.empty(size)
    }

    /// Executes the `kernel` over the given `bindings`.
    pub fn execute(&self, kernel: Server::Kernel, bindings: Vec<Binding<Server>>) {
        self.channel.execute(kernel, bindings)
    }

    /// Wait for the completion of every task in the server.
    pub fn sync(&self) {
        self.channel.sync()
    }

    /// Executes the fastest kernel in the autotune operation, using (cached) runtime benchmarks
    pub fn autotune_execute(
        &self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<Server::AutotuneKey>>,
    ) {
        self.tuner
            .write()
            .unwrap()
            .execute_autotune(autotune_operation_set, self);
    }

    /// Get the fastest kernel for the given autotune key if it exists.
    pub fn autotune_result(&self, key: &Server::AutotuneKey) -> Option<usize> {
        self.tuner.read().unwrap().autotune_fastest(key)
    }
}
