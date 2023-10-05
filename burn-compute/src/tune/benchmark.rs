use std::time::{Duration, Instant};

use crate::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

use super::Operation;

/// Results of a benchmark run.
#[derive(Debug)]
pub struct BenchmarkResult {
    durations: Vec<Duration>,
}

impl BenchmarkResult {
    pub fn median_duration(&self) -> Duration {
        let mut sorted = self.durations.clone();
        sorted.sort();
        *sorted.get(sorted.len() / 2).unwrap()
    }
}

/// Benchmark trait.
pub trait Benchmark<O, S, C>
where
    O: Operation<S>,
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    ///
    /// This should not include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self, client: &ComputeClient<S, C>) -> O::Input;
    /// Execute the benchmark and returns the time it took to complete.
    fn execute(&self, args: O::Input);
    /// Number of samples required to have a statistical significance.
    fn num_samples(&self) -> usize {
        10
    }
    /// Name of the benchmark.
    fn name(&self) -> String;
    /// Run the benchmark a number of times.
    fn run(&self, client: &ComputeClient<S, C>) -> BenchmarkResult {
        // Warmup
        self.execute(self.prepare(client));
        client.sync();

        let mut durations = Vec::with_capacity(self.num_samples());

        for _ in 0..self.num_samples() {
            // Prepare
            let args = self.prepare(client);
            client.sync();

            // Execute the benchmark
            let start = Instant::now();
            self.execute(args);
            client.sync();
            let end = Instant::now();

            // Register the duration
            durations.push(end - start);
        }

        BenchmarkResult { durations }
    }
}
