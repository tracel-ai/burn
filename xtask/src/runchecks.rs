//! This script is run before a PR is created.
//!
//! It is used to check that the code compiles and passes all tests.
//!
//! It is also used to check that the code is formatted correctly and passes clippy.

use crate::logging::init_logger;
use crate::utils::{format_duration, get_workspaces, WorkspaceMemberType};
use crate::{endgroup, group};
use std::env;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::str;
use std::time::Instant;

// Targets constants
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";

// Handle child process
fn handle_child_process(mut child: Child, error: &str) {
    // Wait for the child process to finish
    let status = child.wait().expect(error);

    // If exit status is not a success, terminate the process with an error
    if !status.success() {
        // Use the exit code associated to a command to terminate the process,
        // if any exit code had been found, use the default value 1
        std::process::exit(status.code().unwrap_or(1));
    }
}

// Run a command
fn run_command(command: &str, args: &[&str], command_error: &str, child_error: &str) {
    // Format command
    info!("{command} {}\n\n", args.join(" "));

    // Run command as child process
    let command = Command::new(command)
        .args(args)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()) // Send stderr directly to terminal
        .spawn()
        .expect(command_error);

    // Handle command child process
    handle_child_process(command, child_error);
}

// Define and run rustup command
fn rustup(command: &str, target: &str) {
    group!("Rustup: {} add {}", command, target);
    run_command(
        "rustup",
        &[command, "add", target],
        "Failed to run rustup",
        "Failed to wait for rustup child process",
    );
    endgroup!();
}

// Define and run a cargo command
fn run_cargo(command: &str, params: Params, error: &str) {
    run_cargo_with_path::<String>(command, params, None, error)
}

// Define and run a cargo command with curr dir
fn run_cargo_with_path<P: AsRef<Path>>(
    command: &str,
    params: Params,
    path: Option<P>,
    error: &str,
) {
    // Print cargo command
    info!("cargo {} {}\n", command, params);

    // Run cargo
    let mut cargo = Command::new("cargo");
    cargo
        .env("CARGO_INCREMENTAL", "0")
        .arg(command)
        .args(params.params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()); // Send stderr directly to terminal

    if let Some(path) = path {
        cargo.current_dir(path);
    }

    let cargo_process = cargo.spawn().expect(error);

    // Handle cargo child process
    handle_child_process(cargo_process, "Failed to wait for cargo child process");
}

// Run cargo build command
fn cargo_build(params: Params) {
    // Run cargo build
    run_cargo(
        "build",
        params + "--color=always",
        "Failed to run cargo build",
    );
}

// Run cargo install command
fn cargo_install(params: Params) {
    // Run cargo install
    run_cargo(
        "install",
        params + "--color=always",
        "Failed to run cargo install",
    );
}

// Run cargo test command
fn cargo_test(params: Params) {
    // Run cargo test
    run_cargo(
        "test",
        params + "--color=always" + "--" + "--color=always",
        "Failed to run cargo test",
    );
}

// Run cargo fmt command
fn cargo_fmt() {
    group!("Cargo: fmt");
    run_cargo(
        "fmt",
        ["--check", "--all", "--", "--color=always"].into(),
        "Failed to run cargo fmt",
    );
    endgroup!();
}

// Run cargo clippy command
fn cargo_clippy() {
    if std::env::var("CI").is_ok() {
        return;
    }
    // Run cargo clippy
    run_cargo(
        "clippy",
        ["--color=always", "--all-targets", "--", "-D", "warnings"].into(),
        "Failed to run cargo clippy",
    );
}

// Run cargo doc command
fn cargo_doc(params: Params) {
    // Run cargo doc
    run_cargo("doc", params + "--color=always", "Failed to run cargo doc");
}

// Build and test a crate in a no_std environment
fn build_and_test_no_std<const N: usize>(crate_name: &str, extra_args: [&str; N]) {
    group!("Checks: {} (no-std)", crate_name);

    // Run cargo build --no-default-features
    cargo_build(Params::from(["-p", crate_name, "--no-default-features"]) + extra_args);

    // Run cargo test --no-default-features
    cargo_test(Params::from(["-p", crate_name, "--no-default-features"]) + extra_args);

    // Run cargo build --no-default-features --target wasm32-unknown-unknowns
    cargo_build(
        Params::from([
            "-p",
            crate_name,
            "--no-default-features",
            "--target",
            WASM32_TARGET,
        ]) + extra_args,
    );

    // Run cargo build --no-default-features --target thumbv7m-none-eabi
    cargo_build(
        Params::from([
            "-p",
            crate_name,
            "--no-default-features",
            "--target",
            ARM_TARGET,
        ]) + extra_args,
    );

    endgroup!();
}

// Setup code coverage
fn setup_coverage() {
    // Install llvm-tools-preview
    rustup("component", "llvm-tools-preview");

    // Set coverage environment variables
    env::set_var("RUSTFLAGS", "-Cinstrument-coverage");
    env::set_var("LLVM_PROFILE_FILE", "burn-%p-%m.profraw");
}

// Run grcov to produce lcov.info
fn run_grcov() {
    // grcov arguments
    #[rustfmt::skip]
    let args = [
        ".",
        "--binary-path", "./target/debug/",
        "-s", ".",
        "-t", "lcov",
        "--branch",
        "--ignore-not-existing",
        "--ignore", "/*", // It excludes std library code coverage from analysis
        "--ignore", "xtask/*",
        "--ignore", "examples/*",
        "-o", "lcov.info",
    ];

    run_command(
        "grcov",
        &args,
        "Failed to run grcov",
        "Failed to wait for grcov child process",
    );
}

// Run no_std checks
fn no_std_checks() {
    // Install wasm32 target
    rustup("target", WASM32_TARGET);

    // Install ARM target
    rustup("target", ARM_TARGET);

    // Run checks for the following crates
    build_and_test_no_std("burn", []);
    build_and_test_no_std("burn-core", []);
    build_and_test_no_std(
        "burn-compute",
        ["--features", "channel-mutex storage-bytes"],
    );
    build_and_test_no_std("burn-common", []);
    build_and_test_no_std("burn-tensor", []);
    build_and_test_no_std("burn-ndarray", []);
    build_and_test_no_std("burn-no-std-tests", []);
}

// Test burn-core with tch and wgpu backend
fn burn_core_std() {
    // Run cargo test --features test-tch
    group!("Test: burn-core (tch)");
    cargo_test(["-p", "burn-core", "--features", "test-tch"].into());
    endgroup!();

    // Run cargo test --features test-wgpu
    if std::env::var("DISABLE_WGPU").is_err() {
        group!("Test: burn-core (wgpu)");
        cargo_test(["-p", "burn-core", "--features", "test-wgpu"].into());
        endgroup!();
    }
}

// Test burn-dataset features
fn burn_dataset_features_std() {
    group!("Checks: burn-dataset (all-features)");

    // Run cargo build --all-features
    cargo_build(["-p", "burn-dataset", "--all-features"].into());

    // Run cargo test --all-features
    cargo_test(["-p", "burn-dataset", "--all-features"].into());

    // Run cargo doc --all-features
    cargo_doc(["-p", "burn-dataset", "--all-features"].into());

    endgroup!();
}

// macOS only checks
#[cfg(target_os = "macos")]
fn macos_checks() {
    // Leverages the macOS Accelerate framework: https://developer.apple.com/documentation/accelerate
    group!("Checks: burn-candle (accelerate)");
    cargo_test(["-p", "burn-candle", "--features", "accelerate"].into());
    endgroup!();

    // Leverages the macOS Accelerate framework: https://developer.apple.com/documentation/accelerate
    group!("Checks: burn-ndarray (accelerate)");
    cargo_test(["-p", "burn-ndarray", "--features", "blas-accelerate"].into());
    endgroup!();
}

fn std_checks() {
    // Set RUSTDOCFLAGS environment variable to treat warnings as errors
    // for the documentation build
    env::set_var("RUSTDOCFLAGS", "-D warnings");

    // Check if COVERAGE environment variable is set
    let is_coverage = std::env::var("COVERAGE").is_ok();
    let disable_wgpu = std::env::var("DISABLE_WGPU").is_ok();

    // Check format
    cargo_fmt();

    // Check clippy lints
    cargo_clippy();

    // Produce documentation for each workspace
    group!("Docs: workspaces");
    cargo_doc(["--workspace"].into());
    endgroup!();

    // Setup code coverage
    if is_coverage {
        setup_coverage();
    }

    // Build & test each workspace
    let workspaces = get_workspaces(WorkspaceMemberType::Crate);
    for workspace in workspaces {
        if disable_wgpu && workspace.name == "burn-wgpu" {
            continue;
        }

        if workspace.name == "burn-tch" {
            continue;
        }

        group!("Checks: {}", workspace.name);
        cargo_build(Params::from(["-p", &workspace.name]));
        cargo_test(Params::from(["-p", &workspace.name]));
        endgroup!();
    }

    // Test burn-candle with accelerate (macOS only)
    #[cfg(target_os = "macos")]
    macos_checks();

    // Test burn-dataset features
    burn_dataset_features_std();

    // Test burn-core with tch and wgpu backend
    burn_core_std();

    // Run grcov and produce lcov.info
    if is_coverage {
        run_grcov();
    }
}

fn check_typos() {
    // This path defines where typos-cl is installed on different
    // operating systems.
    let typos_cli_path = std::env::var("CARGO_HOME")
        .map(|v| std::path::Path::new(&v).join("bin/typos-cli"))
        .unwrap();

    // Do not run cargo install on CI to speed up the computation.
    // Check whether the file has been installed on
    if std::env::var("CI").is_err() && !typos_cli_path.exists() {
        // Install typos-cli
        cargo_install(["typos-cli", "--version", "1.16.5"].into());
    }

    info!("Running typos check \n\n");

    // Run typos command as child process
    let typos = Command::new("typos")
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()) // Send stderr directly to terminal
        .spawn()
        .expect("Failed to run typos");

    // Handle typos child process
    handle_child_process(typos, "Failed to wait for typos child process");
}

fn check_examples() {
    let workspaces = get_workspaces(WorkspaceMemberType::Example);
    for workspace in workspaces {
        if workspace.name == "notebook" {
            continue;
        }

        group!("Checks: Example - {}", workspace.name);
        run_cargo_with_path(
            "check",
            ["--examples"].into(),
            Some(workspace.path),
            "Failed to check example",
        );
        endgroup!();
    }
}

#[derive(clap::ValueEnum, Default, Copy, Clone, PartialEq, Eq)]
pub enum CheckType {
    /// Run all checks.
    #[default]
    All,
    /// Run `std` environment checks
    Std,
    /// Run `no-std` environment checks
    NoStd,
    /// Check for typos
    Typos,
    /// Test the examples
    Examples,
}

pub fn run(env: CheckType) -> anyhow::Result<()> {
    // Setup logger
    init_logger().init();

    // Start time measurement
    let start = Instant::now();

    // The environment can assume ONLY "std", "no_std", "typos", "examples"
    // as values.
    //
    // Depending on the input argument, the respective environment checks
    // are run.
    //
    // If no environment has been passed, run all checks.
    match env {
        CheckType::Std => std_checks(),
        CheckType::NoStd => no_std_checks(),
        CheckType::Typos => check_typos(),
        CheckType::Examples => check_examples(),
        CheckType::All => {
            /* Run all checks */
            check_typos();
            std_checks();
            no_std_checks();
            check_examples();
        }
    }

    // Stop time measurement
    //
    // Compute runtime duration
    let duration = start.elapsed();

    // Print duration
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}

struct Params {
    params: Vec<String>,
}

impl<const N: usize> From<[&str; N]> for Params {
    fn from(value: [&str; N]) -> Self {
        Self {
            params: value.iter().map(|v| v.to_string()).collect(),
        }
    }
}

impl From<&str> for Params {
    fn from(value: &str) -> Self {
        Self {
            params: vec![value.to_string()],
        }
    }
}

impl std::fmt::Display for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.params.join(" ").as_str())
    }
}

impl<Rhs: Into<Params>> std::ops::Add<Rhs> for Params {
    type Output = Params;

    fn add(mut self, rhs: Rhs) -> Self::Output {
        let rhs: Params = rhs.into();
        self.params.extend(rhs.params);
        self
    }
}
