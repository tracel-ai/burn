use crate::server::{Binding, ComputeServer, Handle};
use alloc::vec::Vec;
use burn_common::reader::Reader;

/// The ComputeChannel trait links the ComputeClient to the ComputeServer
/// while ensuring thread-safety
pub trait ComputeChannel<Server: ComputeServer>: Clone + core::fmt::Debug + Send + Sync {
    /// Given a binding, returns owned resource as bytes
    fn read(&self, binding: Binding<Server>) -> Reader<Vec<u8>>;

    /// Given a resource as bytes, stores it and returns the resource handle
    fn create(&self, data: &[u8]) -> Handle<Server>;

    /// Reserves `size` bytes in the storage, and returns a handle over them
    fn empty(&self, size: usize) -> Handle<Server>;

    /// Executes the `kernel` over the given `bindings`.
    fn execute(&self, kernel: Server::Kernel, bindings: Vec<Binding<Server>>);

    /// Wait for the completion of every task in the server.
    fn sync(&self);
}
