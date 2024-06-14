use crate::{
    channel::ComputeChannel,
    server::{Binding, ComputeServer, Handle},
    storage::ComputeStorage,
    tune::{AutotuneOperationSet, Tuner},
};
use alloc::vec::Vec;
use alloc::{boxed::Box, sync::Arc};
use burn_common::stub::RwLock;
use burn_common::{reader::Reader, sync_type::SyncType};

/// The ComputeClient is the entry point to require tasks from the ComputeServer.
/// It should be obtained for a specific device via the Compute struct.
#[derive(Debug)]
pub struct ComputeClient<Server: ComputeServer, Channel> {
    channel: Channel,
    tuner: Arc<RwLock<Tuner<Server::AutotuneKey>>>,
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
    pub fn new(channel: Channel, tuner: Arc<RwLock<Tuner<Server::AutotuneKey>>>) -> Self {
        Self { channel, tuner }
    }

    /// Given a binding, returns owned resource as bytes.
    pub fn read(&self, binding: Binding<Server>) -> Reader<Vec<u8>> {
        self.channel.read(binding)
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        binding: Binding<Server>,
    ) -> <Server::Storage as ComputeStorage>::Resource {
        self.channel.get_resource(binding)
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
    pub fn sync(&self, sync_type: SyncType) {
        self.channel.sync(sync_type)
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
