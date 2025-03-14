use clap::{Parser, Subcommand, ValueEnum};
use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
use std::io;
use std::process::ExitStatus;
use std::sync::{Arc, Mutex};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter};

use crate::burnbenchapp::auth::Tokens;
use crate::persistence::system_info::BenchmarkSystemInfo;

use super::auth::get_tokens;
use super::auth::get_username;
use super::progressbar::RunnerProgressBar;
use super::reports::{BenchmarkCollection, FailedBenchmark};
use super::runner::{CargoRunner, NiceProcessor, OutputProcessor, VerboseProcessor};
use super::USER_BENCHMARK_WEBSITE_URL;

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

    /// Enable verbose mode
    #[clap(short = 'v', long = "verbose")]
    verbose: bool,

    /// Space separated list of backends to include
    #[clap(short = 'B', long = "backends", value_name = "BACKEND BACKEND ...", num_args(1..), required = true)]
    backends: Vec<BackendValues>,

    /// Space separated list of benches to run
    #[clap(short = 'b', long = "benches", value_name = "BENCH BENCH ...", num_args(1..), required = true)]
    benches: Vec<BenchmarkValues>,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
enum BackendValues {
    #[strum(to_string = "all")]
    All,
    #[strum(to_string = "candle-cpu")]
    CandleCpu,
    #[strum(to_string = "candle-cuda")]
    CandleCuda,
    #[strum(to_string = "candle-metal")]
    CandleMetal,
    #[strum(to_string = "cuda")]
    Cuda,
    #[strum(to_string = "cuda-fusion")]
    CudaFusion,
    #[cfg(target_os = "linux")]
    #[strum(to_string = "hip")]
    Hip,
    #[strum(to_string = "ndarray")]
    Ndarray,
    #[strum(to_string = "ndarray-simd")]
    NdarraySimd,
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
    #[strum(to_string = "wgpu-spirv")]
    WgpuSpirv,
    #[strum(to_string = "wgpu-spirv-fusion")]
    WgpuSpirvFusion,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
enum BenchmarkValues {
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
    #[strum(to_string = "matmul-fused")]
    MatmulFused,
    #[strum(to_string = "unary")]
    Unary,
    #[strum(to_string = "max-pool2d")]
    MaxPool2d,
    #[strum(to_string = "resnet50")]
    Resnet50,
    #[strum(to_string = "load-record")]
    LoadRecord,
    #[strum(to_string = "autodiff")]
    Autodiff,
    #[strum(to_string = "conv-transpose2d")]
    ConvTranspose2d,
    #[strum(to_string = "conv-transpose3d")]
    ConvTranspose3d,
    #[strum(to_string = "conv2d")]
    Conv2d,
    #[strum(to_string = "conv3d")]
    Conv3d,
    #[strum(to_string = "reduce")]
    Reduce,
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
    let access_token = tokens.map(|t| t.access_token);
    run_backend_comparison_benchmarks(
        &benches,
        &backends,
        access_token.as_deref(),
        run_args.verbose,
    );
}

fn run_backend_comparison_benchmarks(
    benches: &[BenchmarkValues],
    backends: &[BackendValues],
    token: Option<&str>,
    verbose: bool,
) {
    let mut report_collection = BenchmarkCollection::default();
    let total_count: u64 = (backends.len() * benches.len()).try_into().unwrap();
    let runner_pb: Option<Arc<Mutex<RunnerProgressBar>>> = if verbose {
        None
    } else {
        Some(Arc::new(Mutex::new(RunnerProgressBar::new(total_count))))
    };
    // Iterate through every combination of benchmark and backend
    for bench in benches.iter() {
        for backend in backends.iter() {
            let bench_str = bench.to_string();
            let backend_str = backend.to_string();
            let url = format!("{}benchmarks", super::USER_BENCHMARK_SERVER_URL);

            let status = run_cargo(&bench_str, &backend_str, &url, token, &runner_pb);
            let success = status.unwrap().success();

            if success {
                if let Some(ref pb) = runner_pb {
                    pb.lock().unwrap().succeeded_inc();
                }
            } else {
                if let Some(ref pb) = runner_pb {
                    pb.lock().unwrap().failed_inc();
                }
                report_collection.push_failed_benchmark(FailedBenchmark {
                    bench: bench_str.clone(),
                    backend: backend_str.clone(),
                })
            }
        }
    }
    if let Some(pb) = runner_pb.clone() {
        pb.lock().unwrap().finish();
    }
    println!("{}", report_collection.load_records());
    if let Some(url) = web_results_url(token) {
        println!("📊 Browse results at {}", url);
    }
}

fn run_cargo(
    bench: &str,
    backend: &str,
    url: &str,
    token: Option<&str>,
    progress_bar: &Option<Arc<Mutex<RunnerProgressBar>>>,
) -> io::Result<ExitStatus> {
    let processor: Arc<dyn OutputProcessor> = if let Some(pb) = progress_bar {
        Arc::new(NiceProcessor::new(
            bench.to_string(),
            backend.to_string(),
            pb.clone(),
        ))
    } else {
        Arc::new(VerboseProcessor)
    };
    let mut args = vec![
        "-p",
        "backend-comparison",
        "--bench",
        bench,
        "--features",
        backend,
        "--target-dir",
        super::BENCHMARKS_TARGET_DIR,
    ];
    if let Some(t) = token {
        args.push("--");
        args.push("--sharing-url");
        args.push(url);
        args.push("--sharing-token");
        args.push(t);
    }
    let mut runner = CargoRunner::new(&args, processor);
    runner.run()
}

fn web_results_url(token: Option<&str>) -> Option<String> {
    if let Some(t) = token {
        if let Some(user) = get_username(t) {
            let sysinfo = BenchmarkSystemInfo::new();
            let encoded_os = utf8_percent_encode(&sysinfo.os.name, NON_ALPHANUMERIC).to_string();
            let output = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap();
            let git_hash = String::from_utf8(output.stdout).unwrap().trim().to_string();

            return Some(format!(
                "{}benchmarks/community-benchmarks?user={}&os={}&version1={}&version2={}&search=true",
                USER_BENCHMARK_WEBSITE_URL, user.nickname, encoded_os, git_hash, git_hash
            ));
        }
    }
    None
}
