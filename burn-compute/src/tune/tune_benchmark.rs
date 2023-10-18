use burn_common::benchmark::Benchmark;

use crate::server::ComputeServer;

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<S> {
    server: S,
}

impl<S: ComputeServer> Benchmark for TuneBenchmark<S> {
    type Args;

    fn prepare(&self) -> Self::Args {
        todo!()
    }

    fn execute(&self, args: Self::Args) {
        todo!()
    }

    fn name(&self) -> String {
        "Autotune".to_string()
    }

    fn sync(&self) {
        self.server.sync();
    }
}
