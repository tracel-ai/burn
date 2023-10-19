use crate::server::{ComputeServer, Handle};

use super::{MutBenchmark, Operation};

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<'a, S: ComputeServer> {
    operation: Operation<S>,
    handles: Vec<Handle<S>>,
    server: &'a mut S,
}

impl<'a, S: ComputeServer> MutBenchmark for TuneBenchmark<'a, S> {
    type Args = ();

    fn prepare(&self) -> Self::Args {}

    fn execute(&mut self, _: Self::Args) {
        self.operation.clone().execute(self.handles.clone(), self.server)
    }

    fn name(&self) -> String {
        "Autotune".to_string()
    }

    fn sync(&mut self) {
        self.server.sync();
    }
}
