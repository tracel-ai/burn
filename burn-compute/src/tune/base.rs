use core::marker::PhantomData;
use core::time::Duration;

use burn_common::benchmark::BenchmarkResult;
use spin::Mutex;

use crate::server::ComputeServer;
use crate::server::Handle;
use crate::tune::{BenchmarkPool, Operation};

use super::TuneBenchmark;

#[derive(new)]
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
    pub fn tune(&self, input: O::Input, handles: &[&Handle<S>]) -> S::Kernel {
        let mut benchmark_pool = self.benchmark_pool.lock();

        benchmark_pool
            .try_cache(&input)
            .unwrap_or(self.no_kernel_type_found(&mut benchmark_pool, &input, handles))
    }

    fn no_kernel_type_found(
        &self,
        benchmark_pool: &mut BenchmarkPool<TB, O, S>,
        input: &O::Input,
        handles: &[&Handle<S>],
    ) -> S::Kernel {
        let results: Vec<BenchmarkResult> = benchmark_pool
            .tune_benchmarks
            .iter()
            .map(|benchmark| benchmark.run(handles))
            .collect();
        let best_index = self.find_best(results);
        benchmark_pool.add_to_cache(input, best_index);
        benchmark_pool.get_kernel(best_index)
    }

    fn find_best(&self, results: Vec<BenchmarkResult>) -> usize {
        let mut best_duration = Duration::MAX;
        let mut best_tunable = None;

        for (i, result) in results.into_iter().enumerate() {
            let duration = result.median_duration();

            if duration < best_duration {
                best_duration = duration;
                best_tunable = Some(i);
            }
        }

        best_tunable.expect("At least one kernel needed. ")
    }
}
