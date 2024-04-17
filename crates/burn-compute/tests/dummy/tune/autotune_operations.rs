use std::sync::Arc;

use burn_compute::{client::ComputeClient, server::Binding, tune::AutotuneOperation};
use derive_new::new;

use crate::dummy::{DummyChannel, DummyKernel, DummyServer};

#[derive(new)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct OneKernelAutotuneOperation {
    kernel: Arc<dyn DummyKernel>,
    client: ComputeClient<DummyServer, DummyChannel>,
    shapes: Vec<Vec<usize>>,
    handles: Vec<Binding<DummyServer>>,
}

impl AutotuneOperation for OneKernelAutotuneOperation {
    /// Executes the operation on given handles and server, with the additional parameters
    fn execute(self: Box<Self>) {
        self.client.execute(self.kernel.clone(), self.handles);
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Self {
            kernel: self.kernel.clone(),
            client: self.client.clone(),
            shapes: self.shapes.clone(),
            handles: self.handles.clone(),
        })
    }
}
