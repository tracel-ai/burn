use super::system_info::BenchmarkSystemInfo;
use burn::{
    serde::{de::Visitor, ser::SerializeStruct, Deserialize, Serialize, Serializer},
    tensor::backend::Backend,
};
use burn_common::benchmark::BenchmarkResult;
use dirs;
use reqwest::header::{HeaderMap, ACCEPT, AUTHORIZATION, USER_AGENT};
use serde_json;
use std::time::Duration;
use std::{fs, io::Write};

#[derive(Default, Clone)]
pub struct BenchmarkRecord {
    pub backend: String,
    pub device: String,
    pub feature: String,
    pub system_info: BenchmarkSystemInfo,
    pub results: BenchmarkResult,
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
///      "feature": "feature name",
///      "gitHash": "hash",
///      "max": "duration in microseconds",
///      "mean": "duration in microseconds",
///      "median": "duration in microseconds",
///      "min": "duration in microseconds",
///      "name": "benchmark name",
///      "numSamples": "number of samples",
///      "operation": "operation name",
///      "rawDurations": [{"secs": "number of seconds", "nanos": "number of nanons"}, ...],
///      "shapes": [[shape 1], [shape 2], ...],
///      "systemInfo": { "cpus": ["cpu1", "cpu2", ...], "gpus": ["gpu1", "gpu2", ...]}
///      "timestamp": "timestamp",
///      "variance": "duration in microseconds",
///    },
///    { ... }
/// ]
/// ```
pub fn save<B: Backend>(
    benches: Vec<BenchmarkResult>,
    device: &B::Device,
    feature: &str,
    url: Option<&str>,
    token: Option<&str>,
) -> Result<Vec<BenchmarkRecord>, std::io::Error> {
    let cache_dir = dirs::home_dir()
        .expect("Home directory should exist")
        .join(".cache")
        .join("burn")
        .join("backend-comparison");

    for bench in benches.iter() {
        println!("{bench}");
    }

    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)?;
    }

    let records: Vec<BenchmarkRecord> = benches
        .into_iter()
        .map(|bench| BenchmarkRecord {
            backend: B::name(device).to_string(),
            device: format!("{:?}", device),
            feature: feature.to_string(),
            system_info: BenchmarkSystemInfo::new(),
            results: bench,
        })
        .collect();

    for record in records.clone() {
        let file_name = format!(
            "bench_{}_{}.json",
            record.results.name, record.results.timestamp
        );
        let file_path = cache_dir.join(file_name);
        let file =
            fs::File::create(file_path.clone()).expect("Benchmark file should exist or be created");
        serde_json::to_writer_pretty(file, &record)
            .expect("Benchmark file should be updated with benchmark results");

        // Append the benchmark result filepath in the benchmark_results.tx file of
        // cache folder to be later picked by benchrun
        let benchmark_results_path = cache_dir.join("benchmark_results.txt");
        let mut benchmark_results_file = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(benchmark_results_path)
            .unwrap();
        benchmark_results_file
            .write_all(format!("{}\n", file_path.to_string_lossy()).as_bytes())
            .unwrap();

        if let Some(upload_url) = url {
            upload_record(
                &record,
                token.expect("An auth token should be provided."),
                upload_url,
            );
        }
    }

    Ok(records)
}

fn upload_record(record: &BenchmarkRecord, token: &str, url: &str) {
    println!("Sharing results...");
    let client = reqwest::blocking::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, "burnbench".parse().unwrap());
    headers.insert(ACCEPT, "application/json".parse().unwrap());
    headers.insert(AUTHORIZATION, format!("Bearer {}", token).parse().unwrap());
    // post the benchmark record
    let response = client
        .post(url)
        .headers(headers)
        .json(record)
        .send()
        .expect("Request should be sent successfully.");
    if response.status().is_success() {
        println!("Results shared successfully.");
    } else {
        println!("Failed to share results. Status: {}", response.status());
    }
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
            ("feature", &self.feature),
            ("gitHash", &self.results.git_hash),
            ("max", &self.results.computed.max.as_micros()),
            ("mean", &self.results.computed.mean.as_micros()),
            ("median", &self.results.computed.median.as_micros()),
            ("min", &self.results.computed.min.as_micros()),
            ("name", &self.results.name),
            ("numSamples", &self.results.raw.durations.len()),
            ("options", &self.results.options),
            ("rawDurations", &self.results.raw.durations),
            ("systemInfo", &self.system_info),
            ("shapes", &self.results.shapes),
            ("timestamp", &self.results.timestamp),
            ("variance", &self.results.computed.variance.as_micros())
        )
    }
}

struct BenchmarkRecordVisitor;

impl<'de> Visitor<'de> for BenchmarkRecordVisitor {
    type Value = BenchmarkRecord;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "Serialized Json object of BenchmarkRecord")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: burn::serde::de::MapAccess<'de>,
    {
        let mut br = BenchmarkRecord::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "backend" => br.backend = map.next_value::<String>()?,
                "device" => br.device = map.next_value::<String>()?,
                "feature" => br.feature = map.next_value::<String>()?,
                "gitHash" => br.results.git_hash = map.next_value::<String>()?,
                "name" => br.results.name = map.next_value::<String>()?,
                "max" => {
                    let value = map.next_value::<u64>()?;
                    br.results.computed.max = Duration::from_micros(value);
                }
                "mean" => {
                    let value = map.next_value::<u64>()?;
                    br.results.computed.mean = Duration::from_micros(value);
                }
                "median" => {
                    let value = map.next_value::<u64>()?;
                    br.results.computed.median = Duration::from_micros(value);
                }
                "min" => {
                    let value = map.next_value::<u64>()?;
                    br.results.computed.min = Duration::from_micros(value);
                }
                "numSamples" => _ = map.next_value::<usize>()?,
                "options" => br.results.options = map.next_value::<Option<String>>()?,
                "rawDurations" => br.results.raw.durations = map.next_value::<Vec<Duration>>()?,
                "shapes" => br.results.shapes = map.next_value::<Vec<Vec<usize>>>()?,
                "systemInfo" => br.system_info = map.next_value::<BenchmarkSystemInfo>()?,
                "timestamp" => br.results.timestamp = map.next_value::<u128>()?,
                "variance" => {
                    let value = map.next_value::<u64>()?;
                    br.results.computed.variance = Duration::from_micros(value)
                }
                _ => panic!("Unexpected Key: {}", key),
            }
        }

        Ok(br)
    }
}

impl<'de> Deserialize<'de> for BenchmarkRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: burn::serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(BenchmarkRecordVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_benchmark_result() {
        let sample_result = r#"{
            "backend": "candle",
            "device": "Cuda(0)",
            "gitHash": "02d37011ab4dc773286e5983c09cde61f95ba4b5",
            "feature": "wgpu-fusion",
            "name": "unary",
            "max": 8858,
            "mean": 8629,
            "median": 8592,
            "min": 8506,
            "numSamples": 10,
            "options": null,
            "rawDurations": [
                {
                    "secs": 0,
                    "nanos": 8858583
                },
                {
                    "secs": 0,
                    "nanos": 8719822
                },
                {
                    "secs": 0,
                    "nanos": 8705335
                },
                {
                    "secs": 0,
                    "nanos": 8835636
                },
                {
                    "secs": 0,
                    "nanos": 8592507
                },
                {
                    "secs": 0,
                    "nanos": 8506423
                },
                {
                    "secs": 0,
                    "nanos": 8534337
                },
                {
                    "secs": 0,
                    "nanos": 8506627
                },
                {
                    "secs": 0,
                    "nanos": 8521615
                },
                {
                    "secs": 0,
                    "nanos": 8511474
                }
            ],
            "shapes": [
                [
                    32,
                    512,
                    1024
                ]
            ],
            "timestamp": 1710208069697,
            "variance": 0
        }"#;
        let record = serde_json::from_str::<BenchmarkRecord>(sample_result).unwrap();
        assert!(record.backend == "candle");
        assert!(record.device == "Cuda(0)");
        assert!(record.feature == "wgpu-fusion");
        assert!(record.results.git_hash == "02d37011ab4dc773286e5983c09cde61f95ba4b5");
        assert!(record.results.name == "unary");
        assert!(record.results.computed.max.as_micros() == 8858);
        assert!(record.results.computed.mean.as_micros() == 8629);
        assert!(record.results.computed.median.as_micros() == 8592);
        assert!(record.results.computed.min.as_micros() == 8506);
        assert!(record.results.options.is_none());
        assert!(record.results.shapes == vec![vec![32, 512, 1024]]);
        assert!(record.results.timestamp == 1710208069697);
        assert!(record.results.computed.variance.as_micros() == 0);

        //Check raw durations
        assert!(record.results.raw.durations[0] == Duration::from_nanos(8858583));
        assert!(record.results.raw.durations[1] == Duration::from_nanos(8719822));
        assert!(record.results.raw.durations[2] == Duration::from_nanos(8705335));
        assert!(record.results.raw.durations[3] == Duration::from_nanos(8835636));
        assert!(record.results.raw.durations[4] == Duration::from_nanos(8592507));
        assert!(record.results.raw.durations[5] == Duration::from_nanos(8506423));
        assert!(record.results.raw.durations[6] == Duration::from_nanos(8534337));
        assert!(record.results.raw.durations[7] == Duration::from_nanos(8506627));
        assert!(record.results.raw.durations[8] == Duration::from_nanos(8521615));
        assert!(record.results.raw.durations[9] == Duration::from_nanos(8511474));
    }
}
