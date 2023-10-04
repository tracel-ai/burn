use core::time::Duration;

use burn_tensor::benchmark::BenchmarkResult;
use hashbrown::HashMap;
use spin::Mutex;

use crate::tune::{InputHashable, KernelPool, Operation};
use crate::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

use super::KernelType;

struct Tuner<O, S, C>
where
    O: Operation<S>,
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    client: ComputeClient<S, C>,
    kernel_pools: HashMap<O, Mutex<KernelPool<O, S>>>,
}

impl<O, S, C> Tuner<O, S, C>
where
    O: Operation<S>,
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    pub fn tune(&self, operation: O, input: O::Input) {
        let mut kernel_pool = self
            .kernel_pools
            .get(&operation)
            .expect("Called tune on untunable operation")
            .lock();

        let kernel_type = kernel_pool
            .try_cache(&input)
            .unwrap_or(self.no_kernel_type_found(&mut kernel_pool, &input));

        self.execute_found_kernel(kernel_type, input);
    }

    fn execute_found_kernel(&self, kernel_type: KernelType<S>, input: O::Input) {
        let kernel = kernel_type.to_kernel();
        let handles = input.make_handles();
        self.client.execute(kernel, handles)
    }

    fn no_kernel_type_found(
        &self,
        kernel_pool: &mut KernelPool<O, S>,
        input: &O::Input,
    ) -> KernelType<S> {
        let handles = input.make_handles();
        let results: Vec<BenchmarkResult> = kernel_pool
            .kernel_types
            .iter()
            .map(KernelType::to_kernel)
            .map(|kernel| self.client.bench(kernel, handles))
            .collect();
        self.client.sync();
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
