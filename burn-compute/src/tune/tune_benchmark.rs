use burn_common::benchmark::Benchmark;

use crate::client::ComputeClient;
use crate::server::ComputeServer;
use crate::{channel::ComputeChannel, server::Handle};

use super::AutotuneOperation;
use alloc::string::{String, ToString};

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<S: ComputeServer, C> {
    operation: Box<dyn AutotuneOperation<S>>,
    client: ComputeClient<S, C>,
}

impl<S: ComputeServer, C: ComputeChannel<S>> Benchmark for TuneBenchmark<S, C> {
    type Args = (Box<dyn AutotuneOperation<S>>, Vec<Handle<S>>);

    fn prepare(&self) -> Self::Args {
        (
            self.operation.clone(),
            AutotuneOperation::autotune_handles(self.operation.clone()),
        )
    }

    fn num_samples(&self) -> usize {
        10
    }

    // TODO remove mut (and in burn-common too) ?
    fn execute(&mut self, args: Self::Args) {
        let (operation, handles) = args;

        // Ideally this part is not in execute
        let handle_refs: Vec<&Handle<S>> = handles.iter().collect();
        let handle_array: &[&Handle<S>] = &handle_refs;

        AutotuneOperation::execute_for_autotune(operation, handle_array);
    }

    fn name(&self) -> String {
        "Autotune".to_string()
    }

    fn sync(&mut self) {
        self.client.sync();
    }
}
