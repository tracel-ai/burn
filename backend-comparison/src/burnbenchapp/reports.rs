use colored::*;
use core::fmt;
use std::{
    fmt::Display,
    fs,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use crate::persistence::BenchmarkRecord;

pub(crate) struct FailedBenchmark {
    pub(crate) bench: String,
    pub(crate) backend: String,
}

impl fmt::Display for FailedBenchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Run the benchmark with verbose enabled to see the error:\ncargo run --bin burnbench -- run --benches {} --backends {} --verbose",
            self.bench, self.backend
        )
    }
}

pub(crate) struct BenchmarkCollection {
    failed_benchmarks: Vec<FailedBenchmark>,
    results_file: PathBuf,
    successful_records: Vec<BenchmarkRecord>,
}

impl Default for BenchmarkCollection {
    fn default() -> Self {
        let results_file = dirs::home_dir()
            .expect("Home directory should exist")
            .join(".cache")
            .join("burn")
            .join("backend-comparison")
            .join("benchmark_results.txt");
        fs::remove_file(results_file.clone()).ok();
        Self {
            failed_benchmarks: vec![],
            results_file,
            successful_records: vec![],
        }
    }
}

impl BenchmarkCollection {
    pub(crate) fn push_failed_benchmark(&mut self, benchmark: FailedBenchmark) {
        self.failed_benchmarks.push(benchmark);
    }

    pub(crate) fn load_records(&mut self) -> &mut Self {
        if let Ok(file) = fs::File::open(self.results_file.clone()) {
            let file_reader = BufReader::new(file);
            for file in file_reader.lines() {
                let file_path = file.unwrap();
                if let Ok(br_file) = fs::File::open(file_path.clone()) {
                    let benchmarkrecord =
                        serde_json::from_reader::<_, BenchmarkRecord>(br_file).unwrap();
                    self.successful_records.push(benchmarkrecord)
                } else {
                    println!("Cannot find the benchmark-record file: {}", file_path);
                };
            }
        }
        self
    }
}

impl Display for BenchmarkCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Compute the max length for each column
        let mut max_name_len = "Benchmark".len();
        let mut max_backend_len = "Backend".len();
        let mut max_device_len = "Device".len();
        let mut max_feature_len = "Feature".len();
        for record in self.successful_records.iter() {
            max_name_len = max_name_len.max(record.results.name.len());
            // + 2 because if the added backticks
            max_backend_len = max_backend_len.max(record.backend.len() + 2);
            max_device_len = max_device_len.max(record.device.len());
            max_feature_len = max_feature_len.max(record.feature.len());
        }
        for benchmark in self.failed_benchmarks.iter() {
            max_name_len = max_name_len.max(benchmark.bench.len());
            // + 2 because if the added backticks
            max_backend_len = max_backend_len.max(benchmark.backend.len() + 2);
        }
        // Header
        writeln!(
            f,
            "| {:<width_name$} | {:<width_feature$} | {:<width_backend$} | {:<width_device$} | Median         |\n|{:->width_name$}--|{:->width_feature$}--|{:->width_backend$}--|{:->width_device$}--|----------------|",
            "Benchmark",
            "Feature",
            "Backend",
            "Device",
            "",
            "",
            "",
            "",
            width_name = max_name_len,
            width_feature = max_feature_len,
            width_backend = max_backend_len,
            width_device = max_device_len
        )?;
        // Table entries
        // Successful records
        for record in self.successful_records.iter() {
            writeln!(
                f,
                "| {:<width_name$} | {:<width_feature$} | {:<width_backend$} | {:<width_device$} | {:<15.3?}|",
                record.results.name.green(),
                record.feature.green(),
                format!("`{}`", record.backend).green(),
                record.device.green(),
                record.results.computed.median,
                width_name = max_name_len,
                width_feature = max_feature_len,
                width_backend = max_backend_len,
                width_device = max_device_len
            )?;
        }
        // Failed benchmarks
        for benchmark in self.failed_benchmarks.iter() {
            writeln!(
                f,
                "| {:<width_name$} | {:<width_feature$} | {:<width_backend$} | {:<width_device$} | {:<15}|",
                benchmark.bench.red(),
                "-",
                format!("`{}`", benchmark.backend).red(),
                "-",
                "FAILED",
                width_name = max_name_len,
                width_feature = max_feature_len,
                width_backend = max_backend_len,
                width_device = max_device_len
            )?;
        }
        Ok(())
    }
}
