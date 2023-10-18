use core::time::Duration;

use burn_common::benchmark::BenchmarkResult;
use hashbrown::HashMap;

use crate::server::ComputeServer;
use crate::server::Handle;

use super::AutotuneOperation;
use super::Operation;

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub struct Tuner {
    cache: HashMap<String, usize>,
}

impl Tuner {
    pub fn new() -> Self {
        Tuner {
            cache: HashMap::new(),
        }
    }

    /// Looks for cached kernel for the input or finds one manually, saving the fastest one
    pub fn tune<S: ComputeServer>(
        &mut self,
        autotune_operation: Box<dyn AutotuneOperation<S>>,
        autotune_handles: Vec<Handle<S>>,
    ) -> Operation<S> {
        self.try_cache(&autotune_operation)
            .unwrap_or(self.no_kernel_type_found(autotune_operation, autotune_handles))
    }

    fn no_kernel_type_found<S: ComputeServer>(
        &mut self,
        autotune_operation: Box<dyn AutotuneOperation<S>>,
        autotune_handles: Vec<Handle<S>>,
    ) -> Operation<S> {
        let results = autotune_operation
            .autotunables()
            .into_iter()
            .map(|op| self.run_benchmark(op, &autotune_handles))
            .collect();
        let fastest_index = self.find_fastest(results);
        self.cache.insert(autotune_operation.key(), fastest_index);
        autotune_operation.fastest(fastest_index)
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

    fn run_benchmark<S: ComputeServer>(
        &self,
        operation: Operation<S>,
        handles: &Vec<Handle<S>>,
    ) -> BenchmarkResult {
        // operation.execute(handles);
        todo!()
    }

    fn try_cache<S: ComputeServer>(
        &self,
        autotune_kernel: &Box<dyn AutotuneOperation<S>>,
    ) -> Option<Operation<S>> {
        let index = self.cache.get(&autotune_kernel.key());
        if let Some(&i) = index {
            return Some(autotune_kernel.fastest(i));
        }
        None
    }
}
