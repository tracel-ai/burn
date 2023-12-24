use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Display;
use core::time::Duration;

#[cfg(all(not(target_family = "wasm"), feature = "std"))]
use std::time::Instant;
#[cfg(all(target_family = "wasm", feature = "std"))]
use web_time::Instant;

/// Results of a benchmark run.
#[derive(new, Debug)]
pub struct BenchmarkDurations {
    /// All durations of the run, in the order they were benchmarked
    pub durations: Vec<Duration>,
}

impl BenchmarkDurations {
    /// Returns the median duration among all durations
    pub fn median_duration(&self) -> Duration {
        let mut sorted = self.durations.clone();
        sorted.sort();
        *sorted.get(sorted.len() / 2).unwrap()
    }
    pub(crate) fn mean_duration(&self) -> Duration {
        self.durations.iter().sum::<Duration>() / self.durations.len() as u32
    }
}

impl Display for BenchmarkDurations {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
    fn execute(&self, args: Self::Args);
    /// Number of samples required to have a statistical significance.
    fn num_samples(&self) -> usize {
        10
    }
    /// Name of the benchmark.
    fn name(&self) -> String;
    /// Wait for computations to be over
    fn sync(&self);
    /// Run the benchmark a number of times.
    fn run(&self) -> BenchmarkDurations {
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

            BenchmarkDurations { durations }
        }
    }
}

/// Result of a benchmark run, with metadata
pub struct BenchmarkResult {
    /// Individual results of the run
    pub durations: BenchmarkDurations,
    /// Time just before the run
    pub timestamp: u128,
    /// Git commit hash of the commit in which the run occurred
    pub git_hash: String,
    /// Name of the benchmark, normally with operation name and shapes
    pub name: String,
}

impl Display for BenchmarkResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            format!(
                "
        Timestamp: {}
        Git Hash: {}
        Benchmarking - {}{}
        ",
                self.timestamp, self.git_hash, self.name, self.durations
            )
            .as_str(),
        )
    }
}

#[cfg(feature = "std")]
/// Runs the given benchmark on the device and prints result and information.
pub fn run_benchmark<BM>(benchmark: BM) -> BenchmarkResult
where
    BM: Benchmark,
{
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap().trim().to_string();
    BenchmarkResult {
        timestamp,
        git_hash,
        name: benchmark.name(),
        durations: benchmark.run(),
    }
}
