use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use crate::backend::Backend;

/// Results of a benchmark run.
#[derive(Debug)]
pub struct BenchmarkResult {
    durations: Vec<Duration>,
}

impl BenchmarkResult {
    pub(crate) fn mean_duration(&self) -> Duration {
        self.durations.iter().sum::<Duration>() / self.durations.len() as u32
    }

    #[allow(dead_code)]
    pub(crate) fn median_duration(&self) -> Duration {
        let mut sorted = self.durations.clone();
        sorted.sort();
        *sorted.get(sorted.len() / 2).unwrap()
    }
}

impl Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mean = self.mean_duration();
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
pub trait Benchmark<B: Backend> {
    /// Benchmark arguments.
    type Args;

    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    ///
    /// This should not include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self, device: &B::Device) -> Self::Args;
    /// Execute the benchmark and returns the time it took to complete.
    fn execute(&self, args: Self::Args);
    /// Number of samples required to have a statistical significance.
    fn num_samples(&self) -> usize {
        10
    }
    /// Name of the benchmark.
    fn name(&self) -> String;
    /// Run the benchmark a number of times.
    fn run(&self, device: &B::Device) -> BenchmarkResult {
        // Warmup
        self.execute(self.prepare(device));
        B::sync(device);

        let mut durations = Vec::with_capacity(self.num_samples());

        for _ in 0..self.num_samples() {
            // Prepare
            let args = self.prepare(device);
            B::sync(device);

            // Execute the benchmark
            let start = Instant::now();
            self.execute(args);
            B::sync(device);
            let end = Instant::now();

            // Register the duration
            durations.push(end - start);
        }

        BenchmarkResult { durations }
    }
}

/// Runs the given benchmark on the device and prints result and information.
pub fn run_benchmark<B: Backend, BM: Benchmark<B>>(benchmark: BM, device: &B::Device) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap();
    println!("Timestamp: {}", timestamp);
    println!("Git Hash: {}", str::trim(&git_hash));
    println!("Backend: {}", B::name());
    println!(
        "Benchmarking - {}{}",
        benchmark.name(),
        benchmark.run(device)
    );
}
