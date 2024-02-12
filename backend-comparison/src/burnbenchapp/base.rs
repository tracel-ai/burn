use arboard::Clipboard;
use clap::{Parser, Subcommand, ValueEnum};
use github_device_flow::{self, DeviceFlow};
use std::{
    process::{Command, Stdio},
    thread, time,
};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter};

use crate::burnbenchapp::auth::{get_token_from_cache, is_token_valid};

use super::{
    auth::{save_token, CLIENT_ID},
    App,
};

const FIVE_SECONDS: time::Duration = time::Duration::new(5, 0);

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
    #[clap(short = 'B', long = "backends", value_name = "BACKEND,BACKEND,...", num_args(0..))]
    backends: Vec<BackendValues>,

    /// Space separated list of benches to run
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

pub fn execute() {
    let args = Args::parse();
    match args.command {
        Commands::Auth => command_auth(),
        Commands::List => command_list(),
        Commands::Run(run_args) => command_run(run_args),
    }
}

/// Create an access token from GitHub Burnbench application and store it
/// to be used with the user benchmark backend.
fn command_auth() {
    let mut flow = match DeviceFlow::start(CLIENT_ID, None) {
        Ok(flow) => flow,
        Err(e) => {
            eprintln!("Error authenticating: {}", e);
            return;
        }
    };
    println!("🌐 Please visit for following URL in your browser (CTRL+click if your terminal supports it):");
    println!("\n    {}\n", flow.verification_uri.clone().unwrap());
    let user_code = flow.user_code.clone().unwrap();
    println!("👉 And enter code: {}", &user_code);
    if let Ok(mut clipboard) = Clipboard::new() {
        if clipboard.set_text(user_code).is_ok() {
            println!("📋 Code has been successfully copied to clipboard.")
        };
    };
    // Wait for the minimum allowed interval to poll for authentication update
    // see: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#step-3-app-polls-github-to-check-if-the-user-authorized-the-device
    thread::sleep(FIVE_SECONDS);
    match flow.poll(20) {
        Ok(creds) => {
            save_token(&creds.token);
        }
        Err(e) => eprint!("Authentication error: {}", e),
    };
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
    if run_args.share {
        // Verify if a token is saved
        let token = get_token_from_cache();
        if token.is_none() {
            eprintln!("You need to be authenticated to be able to share benchmark results.");
            eprintln!("Run the command 'burnbench auth' to authenticate.");
            return;
        }
        // TODO refresh the token when it is expired
        // Check for the validity of the saved token
        if !is_token_valid(&token.unwrap()) {
            eprintln!("Your access token is no longer valid.");
            eprintln!("Run the command 'burnbench auth' again to get a new token.");
            return;
        }
    }
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
    if run_args.share {
        println!("Sharing results...");
        // TODO Post the results once backend can verify the GitHub access token
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
