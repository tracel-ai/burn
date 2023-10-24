use crate::{
    server::{ComputeServer, Handle},
    tune::AutotuneOperation,
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use burn_common::reader::Reader;

/// The ComputeChannel trait links the ComputeClient to the ComputeServer
/// while ensuring thread-safety
pub trait ComputeChannel<Server: ComputeServer>: Clone + core::fmt::Debug {
    /// Given a handle, returns owned resource as bytes
    fn read(&self, handle: &Handle<Server>) -> Reader<Vec<u8>>;

    /// Given a resource as bytes, stores it and returns the resource handle
    fn create(&self, data: &[u8]) -> Handle<Server>;

    /// Reserves `size` bytes in the storage, and returns a handle over them
    fn empty(&self, size: usize) -> Handle<Server>;

    /// Executes the `kernel` over the given `handles`.
    fn execute(&self, kernel: Server::Kernel, handles: &[&Handle<Server>]);

    /// Wait for the completion of every task in the server.
    fn sync(&self);

    /// Executes the fastest kernel in the autotune operation, using (cached) runtime benchmarks
    fn execute_autotune(
        &self,
        autotune_kernel: Box<dyn AutotuneOperation<Server>>,
        handles: &[&Handle<Server>],
    );
}
