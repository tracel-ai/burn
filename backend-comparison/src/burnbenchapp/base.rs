use clap::{Parser, Subcommand, ValueEnum};
use std::process::ExitStatus;
use std::{
    fs,
    io::{BufRead, BufReader, Result as ioResult},
    process::{Command, Stdio},
};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter};

use crate::burnbenchapp::auth::Tokens;
use crate::persistence::{BenchmarkCollection, BenchmarkRecord};

use super::auth::get_username;
use super::{auth::get_tokens, App};

/// Base trait to define an application
pub(crate) trait Application {
    fn init(&mut self) {}

    #[allow(unused)]
    fn run(
        &mut self,
        benches: &[BenchmarkValues],
        backends: &[BackendValues],
        token: Option<&str>,
    ) {
    }

    fn cleanup(&mut self) {}
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Authenticate using GitHub
    Auth,
    /// List all available benchmarks and backends
    List,
    /// Runs benchmarks
    Run(RunArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Share the benchmark results by uploading them to Burn servers
    #[clap(short = 's', long = "share")]
    share: bool,

    /// Space separated list of backends to include
    #[clap(short = 'B', long = "backends", value_name = "BACKEND BACKEND ...", num_args(1..), required = true)]
    backends: Vec<BackendValues>,

    /// Space separated list of benches to run
    #[clap(short = 'b', long = "benches", value_name = "BENCH BENCH ...", num_args(1..), required = true)]
    benches: Vec<BenchmarkValues>,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
pub(crate) enum BackendValues {
    #[strum(to_string = "all")]
    All,
    #[strum(to_string = "candle-cpu")]
    CandleCpu,
    #[strum(to_string = "candle-cuda")]
    CandleCuda,
    #[strum(to_string = "candle-metal")]
    CandleMetal,
    #[strum(to_string = "ndarray")]
    Ndarray,
    #[strum(to_string = "ndarray-blas-accelerate")]
    NdarrayBlasAccelerate,
    #[strum(to_string = "ndarray-blas-netlib")]
    NdarrayBlasNetlib,
    #[strum(to_string = "ndarray-blas-openblas")]
    NdarrayBlasOpenblas,
    #[strum(to_string = "tch-cpu")]
    TchCpu,
    #[strum(to_string = "tch-gpu")]
    TchGpu,
    #[strum(to_string = "wgpu")]
    Wgpu,
    #[strum(to_string = "wgpu-fusion")]
    WgpuFusion,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
pub(crate) enum BenchmarkValues {
    #[strum(to_string = "all")]
    All,
    #[strum(to_string = "binary")]
    Binary,
    #[strum(to_string = "custom-gelu")]
    CustomGelu,
    #[strum(to_string = "data")]
    Data,
    #[strum(to_string = "matmul")]
    Matmul,
    #[strum(to_string = "unary")]
    Unary,
    #[strum(to_string = "max-pool2d")]
    MaxPool2d,
}

pub fn execute() {
    let args = Args::parse();
    match args.command {
        Commands::Auth => command_auth(),
        Commands::List => command_list(),
        Commands::Run(run_args) => command_run(run_args),
    }
}

/// Create an access token from GitHub Burnbench application, store it,
/// and display the name of the authenticated user.
fn command_auth() {
    get_tokens()
        .and_then(|t| get_username(&t.access_token))
        .map(|user_info| {
            println!("🔑 Your username is: {}", user_info.nickname);
        })
        .unwrap_or_else(|| {
            println!("Failed to display your username.");
        });
}

fn command_list() {
    println!("Available Backends:");
    for backend in BackendValues::iter() {
        println!("- {}", backend);
    }
    println!("\nAvailable Benchmarks:");
    for bench in BenchmarkValues::iter() {
        println!("- {}", bench);
    }
}

fn command_run(run_args: RunArgs) {
    let mut tokens: Option<Tokens> = None;
    if run_args.share {
        tokens = get_tokens();
    }
    // collect benchmarks and benches to execute
    let mut backends = run_args.backends.clone();
    if backends.contains(&BackendValues::All) {
        backends = BackendValues::iter()
            .filter(|b| b != &BackendValues::All)
            .collect();
    }
    let mut benches = run_args.benches.clone();
    if benches.contains(&BenchmarkValues::All) {
        benches = BenchmarkValues::iter()
            .filter(|b| b != &BenchmarkValues::All)
            .collect();
    }

    let total_combinations = backends.len() * benches.len();
    let mut app = App::new();
    app.init();
    println!("Running {} benchmark(s)...\n", total_combinations);
    let access_token = tokens.map(|t| t.access_token);
    app.run(&benches, &backends, access_token.as_deref());
    app.cleanup();
}

#[allow(unused)] // for tui as this is WIP
pub(crate) fn run_cargo(command: &str, params: &[&str]) -> ioResult<ExitStatus> {
    let mut cargo = Command::new("cargo")
        .arg(command)
        .arg("--color=always")
        .args(params)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("cargo process should run");
    cargo.wait()
}

pub(crate) fn run_backend_comparison_benchmarks(
    benches: &[BenchmarkValues],
    backends: &[BackendValues],
    token: Option<&str>,
) {
    let total_count = backends.len() * benches.len();
    let mut current_index = 0;
    // Prefix and postfix for titles
    let filler = ["="; 10].join("");

    // Delete the file containing file paths to benchmark results, if existing
    let benchmark_results_file = dirs::home_dir()
        .expect("Home directory should exist")
        .join(".cache")
        .join("burn")
        .join("backend-comparison")
        .join("benchmark_results.txt");

    fs::remove_file(benchmark_results_file.clone()).ok();

    // Iterate through every combination of benchmark and backend
    for bench in benches.iter() {
        for backend in backends.iter() {
            let bench_str = bench.to_string();
            let backend_str = backend.to_string();
            current_index += 1;
            println!(
                "{} ({}/{}) Benchmarking {} on {} {}",
                filler, current_index, total_count, bench_str, backend_str, filler
            );
            let url = format!("{}benchmarks", super::USER_BENCHMARK_SERVER_URL);
            let mut args = vec![
                "-p",
                "backend-comparison",
                "--bench",
                &bench_str,
                "--features",
                &backend_str,
                "--target-dir",
                super::BENCHMARKS_TARGET_DIR,
            ];
            if let Some(t) = token {
                args.push("--");
                args.push("--sharing-url");
                args.push(url.as_str());
                args.push("--sharing-token");
                args.push(t);
            }
            let status = run_cargo("bench", &args).unwrap();
            if !status.success() {
                println!(
                    "Benchmark {} didn't run successfully on the backend {}",
                    bench_str, backend_str
                );
                continue;
            }
        }
    }

    // Iterate though each benchmark result file present in backend-comparison/benchmark_results.txt
    // and print them in a single table.
    let mut benchmark_results = BenchmarkCollection::default();
    if let Ok(file) = fs::File::open(benchmark_results_file.clone()) {
        let file_reader = BufReader::new(file);
        for file in file_reader.lines() {
            let file_path = file.unwrap();
            if let Ok(br_file) = fs::File::open(file_path.clone()) {
                let benchmarkrecord =
                    serde_json::from_reader::<_, BenchmarkRecord>(br_file).unwrap();
                benchmark_results.records.push(benchmarkrecord)
            } else {
                println!("Cannot find the benchmark-record file: {}", file_path);
            };
        }
        println!(
            "{} Benchmark Results {}\n\n{}",
            filler, filler, benchmark_results
        );
        fs::remove_file(benchmark_results_file).ok();
    }
}
