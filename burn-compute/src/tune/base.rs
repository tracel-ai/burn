use core::marker::PhantomData;
use core::time::Duration;

use burn_common::benchmark::BenchmarkResult;
use hashbrown::HashMap;

use crate::server::ComputeServer;
use crate::server::Handle;

use super::AutotuneKernel;

/// Use to find and reuse the best kernel for some input
pub struct Tuner<S>
where
    S: ComputeServer,
{
    _server: PhantomData<S>,
    cache: HashMap<String, usize>,
}

impl<S> Tuner<S>
where
    S: ComputeServer,
{
    pub fn new() -> Self {
        Tuner {
            cache: HashMap::new(),
            _server: PhantomData,
        }
    }

    /// Looks for cached kernel for the input or finds one manually, saving the fastest one
    pub fn tune(&mut self, autotune_kernel: Box<dyn AutotuneKernel<S>>) -> S::Kernel {
        self.try_cache(autotune_kernel)
            .unwrap_or(self.no_kernel_type_found(autotune_kernel))
    }

    fn no_kernel_type_found(&mut self, autotune_kernel: Box<dyn AutotuneKernel<S>>) -> S::Kernel {
        let results = autotune_kernel
            .autotune_kernels()
            .iter()
            .map(|kernel| self.run_benchmark(kernel, autotune_kernel.autotune_handles()))
            .collect();
        let fastest_index = self.find_fastest(results);
        self.add_to_cache(autotune_kernel, fastest_index);
        autotune_kernel.fastest_kernel(fastest_index)
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

    fn run_benchmark(&self, kernel: &S::Kernel, handles: &[&Handle<S>]) -> BenchmarkResult {
        todo!()
    }

    fn try_cache(&self, autotune_kernel: Box<dyn AutotuneKernel<S>>) -> Option<S::Kernel> {
        let index = self.cache.get(&autotune_kernel.autotune_key());
        if let Some(&i) = index {
            return Some(autotune_kernel.fastest_kernel(i));
        }
        None
    }

    fn add_to_cache(&mut self, autotune_kernel: Box<dyn AutotuneKernel<S>>, index: usize) {
        self.cache.insert(autotune_kernel.autotune_key(), index);
    }
}
