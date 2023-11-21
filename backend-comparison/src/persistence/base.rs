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

/// Updates the cached backend comparison file with new benchmarks,
/// following this json structure:
///
/// In directory BACKEND_NAME:
///     {
///         BENCHMARK_NAME (OP + SHAPE): {
///             GIT_COMMIT_HASH: {
///                 TIMESTAMP: [
///                     DURATIONS
///                 ]
///             }
///         }
///     }
pub fn persist<B: Backend>(benches: Vec<BenchmarkResult>, device: &B::Device) {
    let cache_file = dirs::home_dir()
        .expect("Could not get home directory")
        .join(".cache")
        .join("backend-comparison")
        .join(format!("{}-{:?}.json", B::name(), device));

    println!("Persisting to {:?}", cache_file);
    save(
        fill_backend_comparison(load(cache_file.clone()), benches),
        cache_file,
    )
}

fn fill_backend_comparison(
    mut benchmark_op_results: BenchmarkOpResults,
    benches: Vec<BenchmarkResult>,
) -> BenchmarkOpResults {
    for bench in benches {
        let mut benchmark_commit_results =
            benchmark_op_results.remove(&bench.name).unwrap_or_default();

        let mut stamped_benchmarks = benchmark_commit_results
            .remove(&bench.git_hash)
            .unwrap_or_default();

        stamped_benchmarks.insert(bench.timestamp, bench.durations.durations);
        benchmark_commit_results.insert(bench.git_hash, stamped_benchmarks);
        benchmark_op_results.insert(bench.name, benchmark_commit_results);
    }

    benchmark_op_results
}

fn load(path: PathBuf) -> BenchmarkOpResults {
    match File::open(path) {
        Ok(file) => {
            serde_json::from_reader(file).expect("Should have parsed to BenchmarkOpResults struct")
        }
        Err(_) => BenchmarkOpResults::new(),
    }
}

fn save(backend_comparison: BenchmarkOpResults, path: PathBuf) {
    if let Some(parent) = path.parent() {
        create_dir_all(parent).expect("Unable to create directory");
    }
    let file = File::create(&path).expect("Unable to create backend comparison file");

    serde_json::to_writer(file, &backend_comparison)
        .expect("Unable to write to backend comparison file");
}
