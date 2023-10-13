use core::marker::PhantomData;
use core::time::Duration;

use burn_common::benchmark::{Benchmark, BenchmarkResult};
use hashbrown::HashMap;
use spin::Mutex;

use crate::server::Handle;
use crate::tune::{InputHashable, KernelPool, Operation};
use crate::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

use super::TuneBenchmark;

#[derive(new)]
pub struct Tuner<TB, O, S>
where
    TB: TuneBenchmark<O, S>,
    O: Operation,
    S: ComputeServer,
{
    kernel_pool: Mutex<KernelPool<TB, O, S>>,
    _server: PhantomData<S>,
}

impl<TB, O, S> Tuner<TB, O, S>
where
    TB: TuneBenchmark<O, S>,
    O: Operation,
    S: ComputeServer,
{
    pub fn tune(&self, input: O::Input, handles: &[&Handle<S>]) -> S::Kernel {
        let mut kernel_pool = self.kernel_pool.lock();

        kernel_pool
            .try_cache(&input)
            .unwrap_or(self.no_kernel_type_found(&mut kernel_pool, &input, handles))
    }

    fn no_kernel_type_found(
        &self,
        kernel_pool: &mut KernelPool<TB, O, S>,
        input: &O::Input,
        handles: &[&Handle<S>],
    ) -> S::Kernel {
        let results: Vec<BenchmarkResult> = kernel_pool
            .tune_benchmarks
            .iter()
            .map(|benchmark| benchmark.run(handles))
            .collect();
        let best_index = self.find_best(results);
        kernel_pool.add_to_cache(input, best_index);
        kernel_pool.get(best_index)
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
