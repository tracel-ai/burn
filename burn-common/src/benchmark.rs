use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;
#[cfg(feature = "std")]
use std::time::Instant;

/// Results of a benchmark run.
#[derive(new, Debug)]
pub struct BenchmarkResult {
    durations: Vec<Duration>,
}

impl BenchmarkResult {
    /// Returns the median duration among all durations
    pub fn median_duration(&self) -> Duration {
        let mut sorted = self.durations.clone();
        sorted.sort();
        *sorted.get(sorted.len() / 2).unwrap()
    }
}

/// Benchmark trait.
pub trait Benchmark {
    /// Benchmark arguments.
    type Args;

    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    ///
    /// This should not include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self) -> Self::Args;
    /// Execute the benchmark and returns the time it took to complete.
    fn execute(&mut self, args: Self::Args);
    /// Number of samples required to have a statistical significance.
    fn num_samples(&self) -> usize {
        10
    }
    /// Name of the benchmark.
    fn name(&self) -> String;
    /// Wait for computations to be over
    fn sync(&mut self);
    /// Run the benchmark a number of times.
    fn run(&mut self) -> BenchmarkResult {
        #[cfg(not(feature = "std"))]
        panic!("Attempting to run benchmark in a no-std environment");

        #[cfg(feature = "std")]
        {
            // Warmup
            self.execute(self.prepare());
            self.sync();

            let mut durations = Vec::with_capacity(self.num_samples());

            for _ in 0..self.num_samples() {
                // Prepare
                let args = self.prepare();
                self.sync();

                // Execute the benchmark
                let start = Instant::now();
                self.execute(args);
                self.sync();
                let end = Instant::now();

                // Register the duration
                durations.push(end - start);
            }

            BenchmarkResult { durations }
        }
    }
}
