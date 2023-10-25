use std::sync::Arc;

use burn_compute::{client::ComputeClient, server::Handle, tune::AutotuneOperation};
use derive_new::new;

use crate::dummy::{DummyChannel, DummyKernel, DummyServer};

#[derive(new)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct OneKernelAutotuneOperation {
    kernel: Arc<dyn DummyKernel>,
    client: ComputeClient<DummyServer, DummyChannel>,
    shapes: Vec<Vec<usize>>,
}

impl AutotuneOperation<DummyServer> for OneKernelAutotuneOperation {
    /// Executes the operation on given handles and server, with the additional parameters
    fn execute(self: Box<Self>, inputs: &[&Handle<DummyServer>]) {
        self.client.execute(self.kernel.clone(), inputs);
    }

    fn clone(&self) -> Box<dyn AutotuneOperation<DummyServer>> {
        Box::new(Self {
            kernel: self.kernel.clone(),
            client: self.client.clone(),
            shapes: self.shapes.clone(),
        })
    }

    fn execute_for_autotune(self: Box<Self>, handles: &[&Handle<DummyServer>]) {
        self.client.execute(self.kernel.clone(), handles);
    }

    fn autotune_handles(self: Box<Self>) -> Vec<Handle<DummyServer>> {
        const ARBITRARY_BYTE: u8 = 12; // small so that squared < 256
        let mut handles = Vec::with_capacity(self.shapes.len());
        for shape in self.shapes {
            let n_bytes: usize = shape.iter().product();
            let data = vec![ARBITRARY_BYTE; n_bytes];
            let handle = self.client.create(&data);
            handles.push(handle)
        }
        handles
    }
}
