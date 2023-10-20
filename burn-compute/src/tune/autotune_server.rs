use core::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;
use burn_common::benchmark::{Benchmark, BenchmarkResult};

use crate::{
    server::{ComputeServer, Handle},
    tune::{AutotuneOperation, Operation, TuneBenchmark, TuneCache},
};

/// Server wrapper with extra capability of autotuning kernels
#[derive(Debug)]
pub(crate) struct AutotuneServer<S> {
    pub server: S,
    pub tune_cache: TuneCache<S>,
}

impl<S: ComputeServer> AutotuneServer<S> {
    pub fn new(server: S) -> Self {
        AutotuneServer {
            server,
            tune_cache: TuneCache::new(),
        }
    }

    pub(crate) fn execute_autotune(
        &mut self,
        autotune_operation: Box<dyn AutotuneOperation<S>>,
        execution_handles: &[&Handle<S>],
    ) {
        let operation = self
            .tune_cache
            .try_cache(&autotune_operation)
            .unwrap_or_else(|| self.autotuning(autotune_operation));

        operation.execute(execution_handles, &mut self.server);
    }

    fn autotuning(&mut self, autotune_operation: Box<dyn AutotuneOperation<S>>) -> Operation<S> {
        // Create input buffers for autotune
        let autotune_handles: Vec<Handle<S>> = autotune_operation
            .inputs()
            .iter()
            .map(|input| self.server.create(input))
            .collect();

        // Run all autotune benchmarks
        let results = autotune_operation
            .autotunables()
            .into_iter()
            .map(|op| self.run_benchmark(op, autotune_handles.clone()))
            .collect();

        // Finds the fastest operation, stores it and returns it
        let fastest_index = self.find_fastest(results);
        self.tune_cache
            .cache_insert(autotune_operation.key(), fastest_index);
        self.tune_cache.try_cache(&autotune_operation).unwrap()
    }

    fn run_benchmark(
        &mut self,
        operation: Operation<S>,
        handles: Vec<Handle<S>>,
    ) -> BenchmarkResult {
        TuneBenchmark::new(operation, handles, &mut self.server).run()
    }

    fn find_fastest(&self, results: Vec<BenchmarkResult>) -> usize {
        let mut smallest_duration = Duration::MAX;
        let mut fastest_tunable = None;

        for (i, result) in results.into_iter().enumerate() {
            let duration = result.median_duration();

            if duration < smallest_duration {
                smallest_duration = duration;
                fastest_tunable = Some(i);
            }
        }

        fastest_tunable.expect("At least one kernel needed. ")
    }
}
