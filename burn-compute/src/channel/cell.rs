use super::ComputeChannel;
use crate::server::{ComputeServer, Handle};
use alloc::sync::Arc;
use alloc::vec::Vec;

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
    server: Arc<core::cell::RefCell<Server>>,
}

unsafe impl<Server> Send for RefCellComputeChannel<Server> {}
unsafe impl<Server> Sync for RefCellComputeChannel<Server> {}

impl<S> Clone for RefCellComputeChannel<S> {
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
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
            server: Arc::new(core::cell::RefCell::new(server)),
        }
    }
}

#[cfg_attr(feature = "async-read", async_trait::async_trait)]
impl<Server> ComputeChannel<Server> for RefCellComputeChannel<Server>
where
    Server: ComputeServer,
{
    #[cfg(not(feature = "async-read"))]
    fn read(&self, handle: &Handle<Server>) -> Vec<u8> {
        let mut server = self.server.borrow_mut();

        server.read(handle)
    }

    #[cfg(feature = "async-read")]
    async fn read(&self, handle: &Handle<Server>) -> Vec<u8> {
        todo!();
        // let mut server = self.server.borrow_mut();

        // server.read(handle).await
    }

    fn create(&self, resource: &[u8]) -> Handle<Server> {
        self.server.borrow_mut().create(resource)
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        self.server.borrow_mut().empty(size)
    }

    fn execute(&self, kernel_description: Server::Kernel, handles: &[&Handle<Server>]) {
        self.server
            .borrow_mut()
            .execute(kernel_description, handles)
    }

    fn sync(&self) {
        self.server.borrow_mut().sync()
    }
}
