use core::marker::PhantomData;

/// A local channel with direct connection to the backend runner clients.
pub struct DirectChannel<Backends, Bridge> {
    backends: PhantomData<Backends>,
    bridge: PhantomData<Bridge>,
}

impl<Backends, Bridge> Clone for DirectChannel<Backends, Bridge> {
    fn clone(&self) -> Self {
        Self {
            backends: self.backends,
            bridge: self.bridge,
        }
    }
}
