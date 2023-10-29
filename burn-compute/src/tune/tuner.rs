use core::marker::PhantomData;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;
use burn_common::benchmark::{Benchmark, BenchmarkResult};

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use crate::tune::{AutotuneOperation, AutotuneOperationSet, TuneBenchmark, TuneCache};

#[derive(Debug, Default)]
/// Executes autotune benchmarking and caching
pub struct Tuner<S, C> {
    tune_cache: TuneCache<S>,
    _server: PhantomData<S>,
    _channel: PhantomData<C>,
}

impl<S: ComputeServer, C: ComputeChannel<S>> Tuner<S, C> {
    /// Returns a tuner with empty cache
    pub fn new() -> Self {
        Self {
            tune_cache: TuneCache::new(),
            _server: PhantomData,
            _channel: PhantomData,
        }
    }

    pub(crate) fn execute_autotune(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet>,
        client: &ComputeClient<S, C>,
    ) {
        let operation = match self.tune_cache.try_cache(autotune_operation_set) {
            super::TuneCacheResult::Hit(ops) => ops,
            super::TuneCacheResult::Miss(set) => self.autotuning(set, client),
        };

        AutotuneOperation::execute(operation);
    }

    fn autotuning(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet>,
        client: &ComputeClient<S, C>,
    ) -> Box<dyn AutotuneOperation> {
        let key = autotune_operation_set.key();
        let autotunables = autotune_operation_set.autotunables();
        let mut names = Vec::with_capacity(autotunables.len());

        // Run all autotune benchmarks
        let results: Vec<BenchmarkResult> = autotunables
            .into_iter()
            .map(|op| {
                names.push(format!("{}", op.name()));
                self.run_benchmark(op, client)
            })
            .collect();

        for (name, result) in names.iter().zip(results.iter()) {
            log::info!("Benchmark result {name}-{key} => {result}");
        }

        // Finds the fastest operation, stores it and returns it
        let fastest_index = self.find_fastest(results);
        let fastest_name = names.get(fastest_index).unwrap();
        log::info!("Fastest result {fastest_name}-{key}");

        self.tune_cache.cache_insert(key, fastest_index);
        match self.tune_cache.try_cache(autotune_operation_set) {
            super::TuneCacheResult::Hit(ops) => ops,
            super::TuneCacheResult::Miss(_) => panic!("We just inserted, should not miss"),
        }
    }

    fn run_benchmark(
        &mut self,
        operation: Box<dyn AutotuneOperation>,
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
