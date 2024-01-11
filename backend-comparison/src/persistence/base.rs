use std::fs;

use burn::{
    serde::{ser::SerializeStruct, Serialize, Serializer},
    tensor::backend::Backend,
};
use burn_common::benchmark::BenchmarkResult;
use dirs;
use serde_json;
use uuid::Uuid;

#[derive(Default)]
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
    name: &str,
    benches: Vec<BenchmarkResult>,
    device: &B::Device,
) -> Result<Vec<BenchmarkRecord>, std::io::Error> {
    let cache_dir = dirs::home_dir()
        .expect("Could not get home directory")
        .join(".cache")
        .join("backend-comparison");

    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)?;
    }

    let uuid = Uuid::new_v4().simple().to_string();
    let short_uuid = &uuid[..8];
    let file_name = format!("benchmarks_{}_{}.json", name, short_uuid);
    let file_path = cache_dir.join(file_name);

    let records: Vec<BenchmarkRecord> = benches
        .into_iter()
        .map(|bench| BenchmarkRecord {
            backend: B::name().to_string(),
            device: format!("{:?}", device),
            results: bench,
        })
        .collect();

    let file = fs::File::create(file_path).expect("Unable to create backend comparison file.");
    serde_json::to_writer_pretty(file, &records).expect("Unable to save benchmark results.");

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
            ("rawDurations", &self.results.raw.durations),
            ("numSamples", &self.results.raw.durations.len()),
            ("mean", &self.results.computed.mean.as_secs_f64()),
            ("median", &self.results.computed.median.as_secs_f64()),
            ("variance", &self.results.computed.variance.as_secs_f64()),
            ("min", &self.results.computed.min.as_secs_f64()),
            ("max", &self.results.computed.max.as_secs_f64()),
            ("gitHash", &self.results.git_hash),
            ("name", &self.results.name),
            ("operation", &self.results.operation),
            ("shapes", &self.results.shapes),
            ("timestamp", &self.results.timestamp)
        )
    }
}
