use std::{
    collections::HashMap,
    fs::{create_dir_all, File},
    path::PathBuf,
    time::Duration,
};

use burn::tensor::backend::Backend;
use burn_common::benchmark::BenchmarkResult;
use dirs;
use serde_json;

type BenchmarkOpResults = HashMap<String, BenchmarkCommitResults>;
type BenchmarkCommitResults = HashMap<String, StampedBenchmarks>;
type StampedBenchmarks = HashMap<u128, Vec<Duration>>;

#[derive(Default)]
pub struct Persistence {
    results: HashMap<String, BenchmarkOpResults>,
}

impl Persistence {
    /// Updates the cached backend comparison json file with new benchmarks results.
    ///
    /// The file has the following structure:
    ///
    ///  {
    ///    "BACKEND_NAME-DEVICE":
    ///      {
    ///        "BENCHMARK_NAME (OP + SHAPE)": {
    ///          "GIT_COMMIT_HASH": {
    ///            "TIMESTAMP": \[
    ///              DURATIONS
    ///           \]
    ///         }
    ///       }
    ///    }
    ///  }
    pub fn persist<B: Backend>(benches: Vec<BenchmarkResult>, device: &B::Device) {
        for bench in benches.iter() {
            println!("{}", bench);
        }
        let cache_file = dirs::home_dir()
            .expect("Could not get home directory")
            .join(".cache")
            .join("backend-comparison")
            .join("db.json");

        let mut cache = Self::load(&cache_file);
        cache.update::<B>(device, benches);
        cache.save(&cache_file);
        println!("Persisting to {:?}", cache_file);
    }

    /// Load the cache from disk.
    fn load(path: &PathBuf) -> Self {
        let results = match File::open(path) {
            Ok(file) => serde_json::from_reader(file)
                .expect("Should have parsed to BenchmarkOpResults struct"),
            Err(_) => HashMap::default(),
        };

        Self { results }
    }

    /// Save the cache on disk.
    fn save(&self, path: &PathBuf) {
        if let Some(parent) = path.parent() {
            create_dir_all(parent).expect("Unable to create directory");
        }
        let file = File::create(&path).expect("Unable to create backend comparison file");

        serde_json::to_writer_pretty(file, &self.results)
            .expect("Unable to write to backend comparison file");
    }

    /// Update the cache with the given [benchmark results](BenchmarkResult).
    fn update<B: Backend>(&mut self, device: &B::Device, benches: Vec<BenchmarkResult>) {
        let key = format!("{}-{:?}", B::name(), device);
        let mut results_ops = self.results.remove(&key).unwrap_or_default();

        for bench in benches {
            let mut benchmark_commit_results = results_ops.remove(&bench.name).unwrap_or_default();

            let mut stamped_benchmarks = benchmark_commit_results
                .remove(&bench.git_hash)
                .unwrap_or_default();

            stamped_benchmarks.insert(bench.timestamp, bench.durations.durations);
            benchmark_commit_results.insert(bench.git_hash, stamped_benchmarks);
            results_ops.insert(bench.name, benchmark_commit_results);
        }

        self.results.insert(key, results_ops);
    }
}
