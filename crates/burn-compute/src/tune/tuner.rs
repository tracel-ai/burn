#[cfg(target_family = "wasm")]
use web_time::Duration;

#[cfg(not(target_family = "wasm"))]
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use burn_common::benchmark::{Benchmark, BenchmarkComputations, BenchmarkDurations};

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use crate::tune::{AutotuneOperation, AutotuneOperationSet, TuneBenchmark, TuneCache};

use super::AutotuneKey;

#[derive(Debug)]
/// Executes autotune benchmarking and caching
pub struct Tuner<K: AutotuneKey> {
    tune_cache: TuneCache<K>,
}

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Returns a tuner with cache initialized from persistent cache
    pub fn new(name: &str, device_id: &str) -> Self {
        Self {
            tune_cache: TuneCache::new(name, device_id),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    pub fn autotune_fastest(&self, key: &K) -> Option<usize> {
        self.tune_cache.find_fastest(key)
    }

    /// Execute the fastest autotune operation if known, otherwise perform some benchmarks before.
    pub fn execute_autotune<S, C>(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K>>,
        client: &ComputeClient<S, C>,
    ) where
        S: ComputeServer,
        C: ComputeChannel<S>,
    {
        let operation = match self.tune_cache.try_cache(autotune_operation_set) {
            super::TuneCacheResult::Hit(ops) => ops,
            super::TuneCacheResult::Miss(set) => self.autotuning(set, client),
        };

        AutotuneOperation::execute(operation);
    }

    fn autotuning<S, C>(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K>>,
        client: &ComputeClient<S, C>,
    ) -> Box<dyn AutotuneOperation>
    where
        S: ComputeServer,
        C: ComputeChannel<S>,
    {
        let key = autotune_operation_set.key();
        let autotunables = autotune_operation_set.autotunables();
        let mut names = Vec::with_capacity(autotunables.len());

        let results: Vec<BenchmarkDurations> = autotunables
            .into_iter()
            .map(|op| {
                names.push(op.name().to_string());
                self.run_benchmark(op, client)
            })
            .collect();

        // Finds the fastest operation, stores it and returns it
        let fastest_index = self.find_fastest(results);
        let fastest_name = names.get(fastest_index).unwrap();
        log::info!("Fastest result {fastest_name}-{key}");

        self.tune_cache.cache_insert(key.clone(), fastest_index);
        #[cfg(feature = "autotune-persistent-cache")]
        {
            let checksum = autotune_operation_set.compute_checksum();
            self.tune_cache
                .persistent_cache_insert(key, checksum, fastest_index);
            self.tune_cache.save();
        }

        match self.tune_cache.try_cache(autotune_operation_set) {
            super::TuneCacheResult::Hit(ops) => ops,
            super::TuneCacheResult::Miss(_) => panic!("We just inserted, should not miss"),
        }
    }

    fn run_benchmark<S, C>(
        &mut self,
        operation: Box<dyn AutotuneOperation>,
        client: &ComputeClient<S, C>,
    ) -> BenchmarkDurations
    where
        S: ComputeServer,
        C: ComputeChannel<S>,
    {
        TuneBenchmark::new(operation, client.clone()).run()
    }

    fn find_fastest(&self, results: Vec<BenchmarkDurations>) -> usize {
        let mut smallest_duration = Duration::MAX;
        let mut fastest_tunable = None;

        for (i, result) in results.into_iter().enumerate() {
            let computed = BenchmarkComputations::new(&result);

            if computed.median < smallest_duration {
                smallest_duration = computed.median;
                fastest_tunable = Some(i);
            }
        }

        fastest_tunable.expect("At least one kernel needed. ")
    }
}
