use crate::{pool::get_context, GraphicsApi, WgpuDevice};
use std::{
    fmt::Display,
    time::{Duration, Instant},
};

/// Results of a benchmark run.
#[derive(Debug)]
pub struct BenchmarkResult {
    durations: Vec<Duration>,
}

impl Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mean: Duration = self.durations.iter().sum::<Duration>() / self.durations.len() as u32;
        let var = self
            .durations
            .iter()
            .map(|duration| {
                let tmp = duration.as_secs_f64() - mean.as_secs_f64();
                Duration::from_secs_f64(tmp * tmp)
            })
            .sum::<Duration>()
            / self.durations.len() as u32;

        let mut sorted = self.durations.clone();
        sorted.sort();

        let min = sorted.first().unwrap();
        let max = sorted.last().unwrap();
        let median = sorted.get(sorted.len() / 2).unwrap();
        let num_sample = self.durations.len();

        f.write_str(
            format!(
                "
―――――――― Result ―――――――――
  Samples     {num_sample}
  Mean        {mean:.3?}
  Variance    {var:.3?}
  Median      {median:.3?}
  Min         {min:.3?}
  Max         {max:.3?}
―――――――――――――――――――――――――"
            )
            .as_str(),
        )
    }
}

/// Benchmark trait.
pub trait Benchmark<G: GraphicsApi> {
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
    fn execute(&self, args: Self::Args);
    /// Returns the [device](WgpuDevice) used by the benchmark.
    fn device(&self) -> WgpuDevice;
    /// Run the benchmark a number of times.
    fn run(&self, num_times: usize) -> BenchmarkResult
    where
        Self: Sized,
    {
        let device = self.device();
        let context = get_context::<G>(&device);

        // Warmup
        self.execute(self.prepare());
        context.sync();

        let mut durations = Vec::with_capacity(num_times);

        for _ in 0..num_times {
            // Prepare
            let args = self.prepare();
            context.sync();

            // Execute the benchmark
            let start = Instant::now();
            self.execute(args);
            context.sync();
            let end = Instant::now();

            // Register the duration
            durations.push(end - start);
        }

        BenchmarkResult { durations }
    }
}
