use std::time::{Duration, Instant};

use crate::{pool::get_context, GraphicsApi, WgpuDevice};

/// Benchmark trait.
pub trait Benchmark<G: GraphicsApi> {
    /// Benchmark arguments.
    type Args;

    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    ///
    /// This should be include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self) -> Self::Args;
    /// Execute the benchmark and returns the time it took to complete.
    fn execute(&self, args: Self::Args);
    /// Returns the [device](Device) used by the benchmark.
    fn device(&self) -> WgpuDevice;
    /// Run the benchmark a number of times and returns the durations.
    fn run(&self, num_times: usize) -> Vec<Duration>
    where
        Self: Sized,
    {
        run(self, num_times)
    }
}

/// Run a [benchmark](Benchmark) a number of times and returns the durations.
pub fn run<G, B>(benchmark: &B, num_times: usize) -> Vec<Duration>
where
    G: GraphicsApi,
    B: Benchmark<G>,
{
    let device = benchmark.device();
    let context = get_context::<G>(&device);

    // Warmup
    let args = benchmark.prepare();
    benchmark.execute(args);
    context.sync();

    let mut durations = Vec::with_capacity(num_times);

    for _ in 0..num_times {
        // Prepare
        let args = benchmark.prepare();
        context.sync();

        // Execute the benchmark
        let start = Instant::now();
        benchmark.execute(args);
        context.sync();
        let end = Instant::now();
        durations.push(end - start);
    }

    durations
}
