use std::collections::HashMap;

use serde::{Deserialize, Serialize};

const BENCHMARK_CACHE: &str = "/.cache/burn_bencharks/backend-comparison.json";

#[derive(Serialize, Deserialize)]
struct BackendComparison {
    benchmarks: HashMap<BenchmarkKey, BenchmarkResult>,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    configurations: HashMap<ConfigKey, BenchmarkInstanceResult>,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkInstanceResult {
    timestamp: String,
    commit: String,
}
