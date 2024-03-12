use burn::{
    serde::{de::Visitor, ser::SerializeStruct, Deserialize, Serialize, Serializer},
    tensor::backend::Backend,
};
use burn_common::benchmark::BenchmarkResult;
use dirs;
use reqwest::header::{HeaderMap, ACCEPT, AUTHORIZATION, USER_AGENT};
use serde_json;
use std::time::Duration;
use std::{env, fmt::Display};
use std::{fs, io::Write};

#[derive(Default, Clone)]
pub struct BenchmarkRecord {
    backend: String,
    device: String,
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
        let file =
            fs::File::create(file_path.clone()).expect("Benchmark file should exist or be created");
        serde_json::to_writer_pretty(file, &record)
            .expect("Benchmark file should be updated with benchmark results");

        // Append the benchmark result filepath in a temp file to be later picked by benchrun
        let curdir_filepath = env::current_dir()
            .expect("Cannot resolve current directory")
            .join("benchmark_results.txt");
        let mut curdir_file = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(curdir_filepath)
            .unwrap();
        curdir_file
            .write(format!("{}\n", file_path.to_string_lossy()).as_bytes())
            .unwrap();
        if url.is_some() {
            println!("Sharing results...");
            let client = reqwest::blocking::Client::new();
            let mut headers = HeaderMap::new();
            headers.insert(USER_AGENT, "burnbench".parse().unwrap());
            headers.insert(ACCEPT, "application/json".parse().unwrap());
            headers.insert(
                AUTHORIZATION,
                format!(
                    "Bearer {}",
                    token.expect("An auth token should be provided.")
                )
                .parse()
                .unwrap(),
            );
            // post the benchmark record
            let response = client
                .post(url.expect("A benchmark server URL should be provided."))
                .headers(headers)
                .json(&record)
                .send()
                .expect("Request should be sent successfully.");
            if response.status().is_success() {
                println!("Results shared successfully.");
            } else {
                println!("Failed to share results. Status: {}", response.status());
            }
        }
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
            ("numSamples", &self.results.raw.durations.len()),
            ("options", &self.results.options),
            ("rawDurations", &self.results.raw.durations),
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
                "options" => br.results.options = map.next_value::<Option<String>>()?,
                "rawDurations" => br.results.raw.durations = map.next_value::<Vec<Duration>>()?,
                "shapes" => br.results.shapes = map.next_value::<Vec<Vec<usize>>>()?,
                "timestamp" => br.results.timestamp = map.next_value::<u128>()?,
                "variance" => {
                    let value = map.next_value::<u64>()?;
                    br.results.computed.variance = Duration::from_micros(value)
                }

                "numSamples" => _ = map.next_value::<usize>()?,
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

#[derive(Default)]
pub(crate) struct BenchMarkCollection {
    pub records: Vec<BenchmarkRecord>,
}

impl Display for BenchMarkCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = format!(
            "||{0:<15}||{1:<35}||{2:<15}||\n||{3:=<15}||{4:=<35}||{5:=<15}||",
            "Benchmark", "Backend", "Runtime", "", "", ""
        );

        let rows = self
            .records
            .iter()
            .map(|x| {
                let backend = format!("{}-{}", x.backend, x.device);
                format!(
                    "||{0:<15}||{1:<35}||{2:<15.3?}||\n",
                    x.results.name, backend, x.results.computed.mean
                )
            })
            .collect::<String>();
        write!(f, "{}\n{}", header, rows)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_benchmark_result() {
        let sample_result = r#"{"backend":"candle","device":"Cuda(0)","gitHash":"02d37011ab4dc773286e5983c09cde61f95ba4b5","name":"unary","max":8858,"mean":8629,"median":8592,"min":8506,"numSamples":10,"options":null,"rawDurations":[{"secs":0,"nanos":8858583},{"secs":0,"nanos":8719822},{"secs":0,"nanos":8705335},{"secs":0,"nanos":8835636},{"secs":0,"nanos":8592507},{"secs":0,"nanos":8506423},{"secs":0,"nanos":8534337},{"secs":0,"nanos":8506627},{"secs":0,"nanos":8521615},{"secs":0,"nanos":8511474}],"shapes":[[32,512,1024]],"timestamp":1710208069697,"variance":0}"#;
        let record = serde_json::from_str::<BenchmarkRecord>(sample_result).unwrap();
        assert!(record.backend == "candle");
        assert!(record.device == "Cuda(0)");
        assert!(record.results.git_hash == "02d37011ab4dc773286e5983c09cde61f95ba4b5");
        assert!(record.results.name == "unary");
        assert!(record.results.computed.max.as_micros() == 8858);
        assert!(record.results.computed.mean.as_micros() == 8629);
        assert!(record.results.computed.median.as_micros() == 8592);
        assert!(record.results.computed.min.as_micros() == 8506);
        assert!(record.results.options == None);
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
