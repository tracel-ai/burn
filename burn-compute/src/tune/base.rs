use core::marker::PhantomData;
use core::time::Duration;

use burn_common::benchmark::BenchmarkResult;
use hashbrown::HashMap;

use crate::server::ComputeServer;
use crate::server::Handle;

use super::AutotuneOperation;
use super::Operation;

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub struct Tuner<S> {
    cache: HashMap<String, usize>,
    _server: PhantomData<S>,
}

impl<S: ComputeServer> Tuner<S> {
    pub fn new() -> Self {
        Tuner {
            cache: HashMap::new(),
            _server: PhantomData,
        }
    }

    /// Looks for cached kernel for the input or finds one manually, saving the fastest one
    // pub fn tune(
    //     &mut self,
    //     autotune_operation: Box<dyn AutotuneOperation<S>>,
    //     autotune_handles: Vec<Handle<S>>,
    // ) -> Operation<S> {
    //     self.try_cache(&autotune_operation)
    //         .unwrap_or(self.no_kernel_type_found(autotune_operation, autotune_handles))
    // }

    // pub fn find_fastest_operation(
    //     &mut self,
    //     autotune_operation: Box<dyn AutotuneOperation<S>>,
    //     autotune_handles: Vec<Handle<S>>,
    // ) -> Operation<S> {
    // }

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
