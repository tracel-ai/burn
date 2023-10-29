use burn_common::benchmark::Benchmark;

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;

use super::AutotuneOperation;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<S: ComputeServer, C> {
    operation: Box<dyn AutotuneOperation>,
    client: ComputeClient<S, C>,
}

impl<S: ComputeServer, C: ComputeChannel<S>> Benchmark for TuneBenchmark<S, C> {
    // list of operations
    type Args = Vec<Box<dyn AutotuneOperation>>;

    fn prepare(&self) -> Self::Args {
        vec![self.operation.clone()]
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, args: Self::Args) {
        let operation = args[0].clone(); // TODO rm 0

        AutotuneOperation::execute(operation);
    }

    fn name(&self) -> String {
        "Autotune".to_string()
    }

    fn sync(&self) {
        self.client.sync();
    }
}
