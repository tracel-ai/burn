use core::time::Duration;

use spin::Mutex;

use crate::server::ComputeServer;
use crate::server::Handle;
use crate::tune::{BenchmarkPool, Operation};

use super::BenchmarkResult;
use super::TuneBenchmark;

/// Use to find and reuse the best kernel for some input
pub struct Tuner<TB, O, S>
where
    TB: TuneBenchmark<O, S>,
    O: Operation,
    S: ComputeServer,
{
    benchmark_pool: Mutex<BenchmarkPool<TB, O, S>>,
}

impl<TB, O, S> Tuner<TB, O, S>
where
    TB: TuneBenchmark<O, S>,
    O: Operation,
    S: ComputeServer,
{
    /// Create a tuner over tune benchmarks which contain kernels
    pub fn new(benchmarks: Vec<TB>) -> Self {
        Tuner {
            benchmark_pool: Mutex::new(BenchmarkPool::new(benchmarks)),
        }
    }

    /// Looks for cached kernel for the input or finds one manually, saving the fastest one
    pub fn tune(&self, resources: O::Resources, handles: &[&Handle<S>]) -> S::Kernel {
        let mut benchmark_pool = self.benchmark_pool.lock();

        benchmark_pool
            .try_cache(&resources)
            .unwrap_or(self.no_kernel_type_found(&mut benchmark_pool, &resources, handles))
    }

    fn no_kernel_type_found(
        &self,
        benchmark_pool: &mut BenchmarkPool<TB, O, S>,
        resources: &O::Resources,
        handles: &[&Handle<S>],
    ) -> S::Kernel {
        let results = benchmark_pool.run_benchmarks(handles);
        let best_index = self.find_fastest(results);
        benchmark_pool.add_to_cache(resources, best_index);
        benchmark_pool.get_kernel(best_index)
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
