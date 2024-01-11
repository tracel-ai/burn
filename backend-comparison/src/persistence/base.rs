use std::fs;

use burn::{
    serde::{ser::SerializeStruct, Serialize, Serializer},
    tensor::backend::Backend,
};
use burn_common::benchmark::BenchmarkResult;
use dirs;
use serde_json;

#[derive(Default, Clone)]
pub struct BenchmarkRecord {
    backend: String,
    device: String,
    results: BenchmarkResult,
}

/// Save the benchmarks results on disk.
///
/// The structure is flat so that it can be easily queried from a database
/// like MongoDB.
///
/// ```txt
///  [
///    {
///      "backend": "backend name",
///      "device": "device name",
///      "git_hash": "hash",
///      "name": "benchmark name",
///      "operation": "operation name",
///      "shapes": ["shape dimension", "shape dimension", ...],
///      "timestamp": "timestamp",
///      "numSamples": "number of samples",
///      "min": "duration in seconds",
///      "max": "duration in seconds",
///      "median": "duration in seconds",
///      "mean": "duration in seconds",
///      "variance": "duration in seconds"
///      "rawDurations": ["duration 1", "duration 2", ...],
///    },
///    { ... }
/// ]
/// ```
pub fn save<B: Backend>(
    benches: Vec<BenchmarkResult>,
    device: &B::Device,
) -> Result<Vec<BenchmarkRecord>, std::io::Error> {
    let cache_dir = dirs::home_dir()
        .expect("Home directory should exist")
        .join(".cache")
        .join("backend-comparison");

    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)?;
    }

    let records: Vec<BenchmarkRecord> = benches
        .into_iter()
        .map(|bench| BenchmarkRecord {
            backend: B::name().to_string(),
            device: format!("{:?}", device),
            results: bench,
        })
        .collect();

    for record in records.clone() {
        let file_name = format!(
            "bench_{}_{}.json",
            record.results.name, record.results.timestamp
        );
        let file_path = cache_dir.join(file_name);
        let file = fs::File::create(file_path).expect("Benchmark file should exist or be created");
        serde_json::to_writer_pretty(file, &record)
            .expect("Benchmark file should be updated with benchmark results");
    }

    Ok(records)
}

/// Macro to easily serialize each field in a flatten manner.
/// This macro automatically computes the number of fields to serialize
/// and allows specifying a custom serialization key for each field.
macro_rules! serialize_fields {
    ($serializer:expr, $record:expr, $(($key:expr, $field:expr)),*) => {{
        // Hacky way to get the fields count
        let fields_count = [ $(stringify!($key),)+ ].len();
        let mut state = $serializer.serialize_struct("BenchmarkRecord", fields_count)?;
        $(
            state.serialize_field($key, $field)?;
        )*
            state.end()
    }};
}

impl Serialize for BenchmarkRecord {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_fields!(
            serializer,
            self,
            ("backend", &self.backend),
            ("device", &self.device),
            ("gitHash", &self.results.git_hash),
            ("max", &self.results.computed.max.as_micros()),
            ("mean", &self.results.computed.mean.as_micros()),
            ("median", &self.results.computed.median.as_micros()),
            ("min", &self.results.computed.min.as_micros()),
            ("name", &self.results.name),
            ("numRepeats", &self.results.num_repeats),
            ("numSamples", &self.results.raw.durations.len()),
            ("options", &self.results.options),
            ("rawDurations", &self.results.raw.durations),
            ("shapes", &self.results.shapes),
            ("timestamp", &self.results.timestamp),
            ("variance", &self.results.computed.variance.as_micros())
        )
    }
}
