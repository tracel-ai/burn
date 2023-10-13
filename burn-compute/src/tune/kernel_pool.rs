use burn_common::benchmark::{Benchmark, BenchmarkResult};
use core::marker::PhantomData;
use hashbrown::HashMap;
use std::time::Instant;

use crate::server::{ComputeServer, Handle};

use super::{InputHashable, Operation};

pub trait TuneBenchmark<O: Operation, S: ComputeServer> {
    type Args;
    fn prepare(&self) -> Self::Args;
    fn num_samples(&self) -> usize {
        10
    }

    fn sync(&self);
    fn take_kernel(&self) -> S::Kernel;

    fn execute_with_handles(&self, args: Self::Args, handles: &[&Handle<S>]);

    // useless, replaced by execute_with_handles
    fn execute(&self, args: Self::Args) {}

    fn run(&self, handles: &[&Handle<S>]) -> BenchmarkResult {
        // Warmup
        self.execute_with_handles(self.prepare(), handles);
        self.sync();

        let mut durations = Vec::with_capacity(self.num_samples());

        for _ in 0..self.num_samples() {
            // Prepare
            let args = self.prepare();
            self.sync();

            // Execute the benchmark
            let start = Instant::now();
            self.execute_with_handles(args, handles);
            self.sync();
            let end = Instant::now();

            // Register the duration
            durations.push(end - start);
        }

        BenchmarkResult::new(durations)
    }
}

#[derive(new)]
pub struct KernelPool<TB, O, S> {
    cache: HashMap<String, usize>,
    pub tune_benchmarks: Vec<TB>,
    _operation: PhantomData<O>,
    _server: PhantomData<S>,
}

impl<TB: TuneBenchmark<O, S>, O: Operation, S: ComputeServer> KernelPool<TB, O, S> {
    pub(crate) fn try_cache(&self, input: &O::Input) -> Option<S::Kernel> {
        let index = self.cache.get(&input.custom_hash());
        if let Some(&i) = index {
            return Some(self.tune_benchmarks[i].take_kernel());
        }
        None
    }

    pub(crate) fn get(&self, index: usize) -> S::Kernel {
        (*self.tune_benchmarks.get(index).unwrap()).take_kernel()
    }

    pub(crate) fn add_to_cache(&mut self, input: &O::Input, index: usize) -> () {
        self.cache.insert(input.custom_hash(), index);
    }
}
