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

impl Serialize for BenchmarkRecord {
    /// Flatten all the fields when serializing, i.e. we remove the nesting
    /// under "results" and "computed".
    /// Also format the fields to be compliant with MongoDB naming conventions.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("BenchmarkRecord", 11)?;
        state.serialize_field("backend", &self.backend)?;
        state.serialize_field("device", &self.device)?;
        // Serialize fields of BenchmarkResult
        state.serialize_field("rawDurations", &self.results.raw.durations)?;
        state.serialize_field("numSamples", &self.results.raw.durations.len())?;
        state.serialize_field("mean", &self.results.computed.mean.as_secs_f64())?;
        state.serialize_field("median", &self.results.computed.median.as_secs_f64())?;
        state.serialize_field("variance", &self.results.computed.variance.as_secs_f64())?;
        state.serialize_field("min", &self.results.computed.min.as_secs_f64())?;
        state.serialize_field("max", &self.results.computed.max.as_secs_f64())?;
        state.serialize_field("gitHash", &self.results.git_hash)?;
        state.serialize_field("name", &self.results.name)?;
        state.serialize_field("operation", &self.results.operation)?;
        state.serialize_field("shapes", &self.results.shapes)?;
        state.serialize_field("timestamp", &self.results.timestamp)?;
        state.end()
    }
}
