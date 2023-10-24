use super::ComputeChannel;
use crate::server::{ComputeServer, Handle};
use crate::tune::{AutotuneOperation, AutotuneServer};
use alloc::sync::Arc;
use alloc::vec::Vec;
use burn_common::reader::Reader;

/// A channel using a [ref cell](core::cell::RefCell) to access the server with mutability.
///
/// # Important
///
/// Only use this channel if you don't use any threading in your application, otherwise it will
/// panic or cause undefined behaviors.
///
/// This is mosly useful for `no-std` environments where threads aren't supported, otherwise prefer
/// the [mutex](super::MutexComputeChannel) or the [mpsc](super::MpscComputeChannel) channels.
#[derive(Debug)]
pub struct RefCellComputeChannel<Server> {
    autotune_server: Arc<core::cell::RefCell<AutotuneServer<Server>>>,
}

impl<S> Clone for RefCellComputeChannel<S> {
    fn clone(&self) -> Self {
        Self {
            autotune_server: self.autotune_server.clone(),
        }
    }
}
impl<Server> RefCellComputeChannel<Server>
where
    Server: ComputeServer,
{
    /// Create a new cell compute channel.
    pub fn new(server: Server) -> Self {
        Self {
            autotune_server: Arc::new(core::cell::RefCell::new(AutotuneServer::new(server))),
        }
    }
}

impl<Server> ComputeChannel<Server> for RefCellComputeChannel<Server>
where
    Server: ComputeServer,
{
    fn read(&self, handle: &Handle<Server>) -> Reader<Vec<u8>> {
        self.autotune_server.borrow_mut().server.read(handle)
    }

    fn create(&self, resource: &[u8]) -> Handle<Server> {
        self.autotune_server.borrow_mut().server.create(resource)
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        self.autotune_server.borrow_mut().server.empty(size)
    }

    fn execute(&self, kernel_description: Server::Kernel, handles: &[&Handle<Server>]) {
        self.autotune_server
            .borrow_mut()
            .server
            .execute(kernel_description, handles)
    }

    fn sync(&self) {
        self.autotune_server.borrow_mut().server.sync()
    }

    fn execute_autotune(
        &self,
        autotune_kernel: Box<dyn AutotuneOperation<Server>>,
        handles: &[&Handle<Server>],
    ) {
        self.autotune_server
            .borrow_mut()
            .execute_autotune(autotune_kernel, handles);
    }
}
