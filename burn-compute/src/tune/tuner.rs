use core::marker::PhantomData;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;
use burn_common::benchmark::{Benchmark, BenchmarkResult};

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use crate::tune::{AutotuneOperation, AutotuneOperationSet, TuneBenchmark, TuneCache};

/// Server wrapper with extra capability of autotuning kernels
#[derive(Debug)]
pub struct Tuner<S, C> {
    pub tune_cache: TuneCache<S>,
    _server: PhantomData<S>,
    _channel: PhantomData<C>,
}

impl<S: ComputeServer, C: ComputeChannel<S>> Tuner<S, C> {
    pub fn new() -> Self {
        Self {
            tune_cache: TuneCache::new(),
            _server: PhantomData,
            _channel: PhantomData,
        }
    }

    pub(crate) fn execute_autotune(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<S>>,
        client: &ComputeClient<S, C>,
    ) {
        let operation = self
            .tune_cache
            .try_cache(&autotune_operation_set)
            .unwrap_or_else(|| self.autotuning(autotune_operation_set, client));

        AutotuneOperation::execute(operation);
    }

    fn autotuning(
        &mut self,
        autotune_operation: Box<dyn AutotuneOperationSet<S>>,
        client: &ComputeClient<S, C>,
    ) -> Box<dyn AutotuneOperation<S>> {
        // Run all autotune benchmarks
        let results = autotune_operation
            .autotunables()
            .into_iter()
            .map(|op| self.run_benchmark(op, client))
            .collect();

        // Finds the fastest operation, stores it and returns it
        let fastest_index = self.find_fastest(results);
        self.tune_cache
            .cache_insert(autotune_operation.key(), fastest_index);
        self.tune_cache.try_cache(&autotune_operation).unwrap()
    }

    fn run_benchmark(
        &mut self,
        operation: Box<dyn AutotuneOperation<S>>,
        client: &ComputeClient<S, C>,
    ) -> BenchmarkResult {
        TuneBenchmark::new(operation, client.clone()).run()
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
