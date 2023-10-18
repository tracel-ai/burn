use crate::{
    server::{ComputeServer, Handle},
    tune::{AutotuneOperation, Tuner},
};

/// Server with extra capability of autotuning kernels
#[derive(Debug)]
pub(crate) struct AutotuneServer<S> {
    pub server: S,
    pub tuner: Tuner,
}

impl<S: ComputeServer> AutotuneServer<S> {
    pub fn new(server: S) -> Self {
        AutotuneServer {
            server,
            tuner: Tuner::new(),
        }
    }

    pub fn execute_autotune(
        &mut self,
        autotune_kernel: Box<dyn AutotuneOperation<S>>,
        execution_handles: &[&Handle<S>],
    ) {
        let autotune_handles: Vec<Handle<S>> = autotune_kernel
            .inputs()
            .iter()
            .map(|input| self.server.create(input))
            .collect();
        let operation = self.tuner.tune(autotune_kernel, autotune_handles);
        let kernel = operation.get_kernel(); // not sure
        self.server.execute_kernel(kernel, execution_handles);
    }
}
