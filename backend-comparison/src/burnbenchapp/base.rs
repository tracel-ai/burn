use clap::{Parser, Subcommand, ValueEnum};
use std::process::{Command, Stdio};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter};

use super::App;

/// Base trait to define an application
pub(crate) trait Application {
    fn init(&mut self) {}

    #[allow(unused)]
    fn run(&mut self, benches: &[BenchmarkValues], backends: &[BackendValues]) {}

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
    /// List all available benchmarks and backends
    List,
    /// Runs benchmarks
    Run(RunArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Comma-separated list of backends to include
    #[clap(short = 'B', long = "backends", value_name = "BACKEND,BACKEND,...", num_args(0..))]
    backends: Vec<BackendValues>,

    /// Comma-separated list of benches to run
    #[clap(short = 'b', long = "benches", value_name = "BACKEND,BACKEND,...", num_args(0..))]
    benches: Vec<BenchmarkValues>,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
pub(crate) enum BackendValues {
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
    #[strum(to_string = "binary")]
    Binary,
    #[strum(to_string = "custom_gelu")]
    CustomGelu,
    #[strum(to_string = "data")]
    Data,
    #[strum(to_string = "matmul")]
    Matmul,
    #[strum(to_string = "unary")]
    Unary,
}

pub fn run() {
    let args = Args::parse();

    match args.command {
        Commands::List => {
            println!("Available Backends:");
            for backend in BackendValues::iter() {
                println!("- {}", backend);
            }

            println!("\nAvailable Benchmarks:");
            for bench in BenchmarkValues::iter() {
                println!("- {}", bench);
            }
        }
        Commands::Run(run_args) => {
            if run_args.backends.is_empty() || run_args.benches.is_empty() {
                println!("No backends or benchmarks specified. Please select at least one backend and one benchmark.");
                return;
            }

            let total_combinations = run_args.backends.len() * run_args.benches.len();
            println!(
                "Executing the following benchmark and backend combinations (Total: {}):",
                total_combinations
            );
            for backend in &run_args.backends {
                for bench in &run_args.benches {
                    println!("- Benchmark: {}, Backend: {}", bench, backend);
                }
            }

            let mut app = App::new();
            app.init();
            println!("Running benchmarks...");
            app.run(&run_args.benches, &run_args.backends);
            app.cleanup();
            println!("Cleanup completed. Benchmark run(s) finished.");
        }
    }
}

#[allow(unused)] // for tui as this is WIP
pub(crate) fn run_cargo(command: &str, params: &[&str]) {
    let mut cargo = Command::new("cargo")
        .arg(command)
        .arg("--color=always")
        .args(params)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("cargo process should run");
    let status = cargo.wait().expect("");
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}
