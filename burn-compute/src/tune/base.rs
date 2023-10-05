use core::time::Duration;

use burn_common::benchmark::{Benchmark, BenchmarkResult};
use hashbrown::HashMap;
use spin::Mutex;

use crate::tune::{InputHashable, KernelPool, Operation};
use crate::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

use super::{KernelType, TuneBenchmark};

#[derive(new)]
pub struct Tuner<TB, O, S, C>
where
    TB: TuneBenchmark,
    O: Operation<S>,
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    client: ComputeClient<S, C>,
    kernel_pool: Mutex<KernelPool<TB, O, S, C>>,
}

impl<TB, O, S, C> Tuner<TB, O, S, C>
where
    TB: TuneBenchmark,
    O: Operation<S>,
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    pub fn tune(&self, input: O::Input) {
        let mut kernel_pool = self.kernel_pool.lock();

        let kernel = kernel_pool
            .try_cache(&input)
            .unwrap_or(self.no_kernel_type_found(&mut kernel_pool, &input));

        self.execute_found_kernel(kernel, input);
    }

    fn execute_found_kernel(&self, kernel_type: KernelType<TB, O, S, C>, input: O::Input) {
        todo!()
        // let kernel = kernel_type.to_kernel();
        // let handles = input.make_handles();
        // self.client.execute(kernel, handles)
    }

    fn no_kernel_type_found(
        &self,
        kernel_pool: &mut KernelPool<TB, O, S, C>,
        input: &O::Input,
    ) -> KernelType<TB, O, S, C> {
        let results: Vec<BenchmarkResult> = kernel_pool
            .kernel_types
            .iter()
            .map(KernelType::to_benchmark)
            .map(|benchmark| benchmark.run())
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
