use crate::server::{ComputeServer, ExecutionBufferHandle, TensorBufferHandle};
use alloc::vec::Vec;
use burn_common::reader::Reader;

/// The ComputeChannel trait links the ComputeClient to the ComputeServer
/// while ensuring thread-safety
pub trait ComputeChannel<Server: ComputeServer>: Clone + core::fmt::Debug + Send + Sync {
    /// Given a handle, returns owned resource as bytes
    fn read(&self, handle: ExecutionBufferHandle<Server>) -> Reader<Vec<u8>>;

    /// Given a resource as bytes, stores it and returns the resource handle
    fn create(&self, data: &[u8]) -> TensorBufferHandle<Server>;

    /// Reserves `size` bytes in the storage, and returns a handle over them
    fn empty(&self, size: usize) -> TensorBufferHandle<Server>;

    /// Executes the `kernel` over the given `handles`.
    fn execute(&self, kernel: Server::Kernel, handles: Vec<ExecutionBufferHandle<Server>>);

    /// Wait for the completion of every task in the server.
    fn sync(&self);
}
