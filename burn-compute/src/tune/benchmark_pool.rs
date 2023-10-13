use core::{marker::PhantomData, time::Duration};
use hashbrown::HashMap;
use std::time::Instant;

use crate::server::{ComputeServer, Handle};

use super::{HashableResources, Operation};

/// Contains the durations of all samples
#[derive(new, Debug)]
pub struct BenchmarkResult {
    durations: Vec<Duration>,
}

impl BenchmarkResult {
    /// Returns the median duration among all samples
    pub fn median_duration(&self) -> Duration {
        let mut sorted = self.durations.clone();
        sorted.sort();
        *sorted.get(sorted.len() / 2).unwrap()
    }
}

/// A benchmark that runs on server handles
pub trait TuneBenchmark<O: Operation, S: ComputeServer> {
    /// Makes a new instance of the kernel
    fn make_kernel(&self) -> S::Kernel;

    /// Gives how many samples should run
    fn num_samples(&self) -> usize {
        10
    }

    /// Ensures all previous async computations are done
    fn sync(&self);

    /// Executes the kernel on the given handles
    fn execute(&self, kernel: S::Kernel, handles: &[&Handle<S>]);

    /// Runs the benchmark, with kernel creation first
    fn run(&self, handles: &[&Handle<S>]) -> BenchmarkResult {
        // Warmup
        self.execute(self.make_kernel(), handles);
        self.sync();

        let mut durations = Vec::with_capacity(self.num_samples());

        for _ in 0..self.num_samples() {
            // Prepare
            let args = self.make_kernel();
            self.sync();

            // Execute the benchmark
            let start = Instant::now();
            self.execute(args, handles);
            self.sync();
            let end = Instant::now();

            // Register the duration
            durations.push(end - start);
        }

        BenchmarkResult::new(durations)
    }
}

/// A collection of tune benchmarks over the same operation
pub(crate) struct BenchmarkPool<TB, O, S> {
    cache: HashMap<String, usize>,
    tune_benchmarks: Vec<TB>,
    _operation: PhantomData<O>,
    _server: PhantomData<S>,
}

impl<TB: TuneBenchmark<O, S>, O: Operation, S: ComputeServer> BenchmarkPool<TB, O, S> {
    pub(crate) fn new(tune_benchmarks: Vec<TB>) -> Self {
        BenchmarkPool {
            cache: HashMap::new(),
            tune_benchmarks,
            _operation: PhantomData,
            _server: PhantomData,
        }
    }

    pub(crate) fn run_benchmarks(&self, handles: &[&Handle<S>]) -> Vec<BenchmarkResult> {
        self.tune_benchmarks
            .iter()
            .map(|benchmark| benchmark.run(handles))
            .collect()
    }
}

impl<TB: TuneBenchmark<O, S>, O: Operation, S: ComputeServer> BenchmarkPool<TB, O, S> {
    pub(crate) fn try_cache(&self, resources: &O::Resources) -> Option<S::Kernel> {
        let index = self.cache.get(&resources.key());
        if let Some(&i) = index {
            return Some(self.tune_benchmarks[i].make_kernel());
        }
        None
    }

    pub(crate) fn get_kernel(&self, index: usize) -> S::Kernel {
        (*self.tune_benchmarks.get(index).unwrap()).make_kernel()
    }

    pub(crate) fn add_to_cache(&mut self, resources: &O::Resources, index: usize) {
        self.cache.insert(resources.key(), index);
    }
}
