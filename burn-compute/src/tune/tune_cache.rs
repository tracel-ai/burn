use core::marker::PhantomData;
use core::time::Duration;

use burn_common::benchmark::BenchmarkResult;
use hashbrown::HashMap;

use crate::server::ComputeServer;

use super::AutotuneOperation;
use super::Operation;

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub struct TuneCache<S> {
    cache: HashMap<String, usize>,
    _server: PhantomData<S>,
}

impl<S: ComputeServer> TuneCache<S> {
    pub(crate) fn new() -> Self {
        TuneCache {
            cache: HashMap::new(),
            _server: PhantomData,
        }
    }

    pub(crate) fn find_fastest(&self, results: Vec<BenchmarkResult>) -> usize {
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

    pub(crate) fn try_cache(
        &self,
        autotune_operation: &Box<dyn AutotuneOperation<S>>,
    ) -> Option<Operation<S>> {
        let index = self.cache.get(&autotune_operation.key());
        if let Some(&i) = index {
            return Some(autotune_operation.fastest(i));
        }
        None
    }

    pub(crate) fn cache_insert(&mut self, key: String, fastest_index: usize) -> () {
        self.cache.insert(key, fastest_index);
    }
}
