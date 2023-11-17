use std::{collections::HashMap, time::Duration};

use burn::tensor::backend::Backend;
use burn_common::benchmark::BenchmarkResult;
use serde::{Deserialize, Serialize};

const BENCHMARK_CACHE: &str = "/.cache/burn_bencharks/backend-comparison.json";

/// {
///     BACKEND: {
///     BENCHMARK_NAME (op + shape): {
///         COMMIT: {
///                 TIMESTAMP: {
///                     "durations": [12.1, 13.2, 14.2]
///                 }
///             }
///         }
///     }
/// }
///

type BackendName = String;
type OpName = String;
type GitHash = String;
type Timestamp = u128;

type BackendComparison = HashMap<BackendName, BenchmarkOpResults>;
type BenchmarkOpResults = HashMap<OpName, BenchmarkCommitResults>;
type BenchmarkCommitResults = HashMap<GitHash, Durations>;
type StampedBenchmarks = HashMap<Timestamp, Vec<Duration>>;

pub fn persist<B: Backend>(benches: Vec<BenchmarkResult>) {
    let backend_comparison: BackendComparison = load(B);
    let benchmark: Option<BenchmarkCommitResults> = backend_comparison[B::name()];
    for bench in benches {
        let durations = Durations {
            durations: bench.durations.durations,
        };
    }
    BackendComparison { benchmarks: B }
}
