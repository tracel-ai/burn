//! This script is run before a PR is created.
//!
//! It is used to check that the code compiles and passes all tests.
//!
//! It is also used to check that the code is formatted correctly and passes clippy.
//!
//! To build this script, run the following command:
//!
//! rustc scripts/run-checks.rs --crate-type bin --out-dir scripts
//!
//! To run the script:
//!
//! ./scripts/run-checks environment
//!
//! where `environment` can assume **ONLY** the following values:
//!     - `std` to perform checks using `libstd`
//!     - `no_std` to perform checks on an embedded environment using `libcore`
//!     - `typos` to check typos in the source code

use std::env;
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

// Define and run rustup command
fn rustup(target: &str) {
    // Rustup arguments
    let args = ["target", "add", target];

    // Print rustup command
    println!("rustup {}\n\n", args.join(" "));

    // Run rustup command as child process
    let rustup = Command::new("rustup")
        .args(args)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()) // Send stderr directly to terminal
        .spawn()
        .expect("Failed to run rustup");

    // Handle rustup child process
    handle_child_process(rustup, "Failed to wait for rustup child process");
}

// Define and run a cargo command
fn run_cargo(command: &str, first_params: &[&str], second_params: &[&str], error: &str) {
    // Print cargo command
    println!(
        "\ncargo {} {} {}\n",
        command,
        first_params.join(" "),
        second_params.join(" ")
    );

    // Run cargo
    let cargo = Command::new("cargo")
        .arg(command)
        .args(first_params)
        .args(second_params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()) // Send stderr directly to terminal
        .spawn()
        .expect(error);

    // Handle cargo child process
    handle_child_process(cargo, "Failed to wait for cargo child process");
}

// Run cargo build command
fn cargo_build(params: &[&str]) {
    // Run cargo build
    run_cargo(
        "build",
        params,
        &["--color=always"],
        "Failed to run cargo build",
    );
}

// Run cargo install command
fn cargo_install(params: &[&str]) {
    // Run cargo install
    run_cargo(
        "install",
        params,
        &["--color=always"],
        "Failed to run cargo install",
    );
}

// Run cargo test command
fn cargo_test(params: &[&str]) {
    // Run cargo test
    run_cargo(
        "test",
        params,
        &["--color=always", "--", "--color=always"],
        "Failed to run cargo test",
    );
}

// Run cargo fmt command
fn cargo_fmt() {
    // Run cargo fmt
    run_cargo(
        "fmt",
        &["--check", "--all"],
        &["--", "--color=always"],
        "Failed to run cargo fmt",
    );
}

// Run cargo clippy command
fn cargo_clippy() {
    // Run cargo clippy
    run_cargo(
        "clippy",
        &["--color=always"],
        &["--", "-D", "warnings"],
        "Failed to run cargo clippy",
    );
}

// Run cargo doc command
fn cargo_doc(params: &[&str]) {
    // Run cargo doc
    run_cargo(
        "doc",
        params,
        &["--color=always"],
        "Failed to run cargo doc",
    );
}

// Build and test a crate in a no_std environment
fn build_and_test_no_std(crate_name: &str) {
    println!("\nRun checks for `{}` crate", crate_name);

    // Run cargo build --no-default-features
    cargo_build(&["-p", crate_name, "--no-default-features"]);

    // Run cargo test --no-default-features
    cargo_test(&["-p", crate_name, "--no-default-features"]);

    // Run cargo build --no-default-features --target wasm32-unknown-unknowns
    cargo_build(&[
        "-p",
        crate_name,
        "--no-default-features",
        "--target",
        WASM32_TARGET,
    ]);

    // Run cargo build --no-default-features --target thumbv7m-none-eabi
    cargo_build(&[
        "-p",
        crate_name,
        "--no-default-features",
        "--target",
        ARM_TARGET,
    ]);
}

// Run no_std checks
fn no_std_checks() {
    println!("Checks for no_std environment...\n\n");

    // Install wasm32 target
    rustup(WASM32_TARGET);

    // Install ARM target
    rustup(ARM_TARGET);

    // Run checks for the following crates
    build_and_test_no_std("burn");
    build_and_test_no_std("burn-core");
    build_and_test_no_std("burn-common");
    build_and_test_no_std("burn-tensor");
    build_and_test_no_std("burn-ndarray");
    build_and_test_no_std("burn-no-std-tests");
}

// Test burn-core with tch and wgpu backend
fn burn_core_std() {
    println!("\n\nRun checks for burn-core crate with tch and wgpu backend");

    // Run cargo test --features test-tch
    cargo_test(&["-p", "burn-core", "--features", "test-tch"]);

    // Run cargo test --features test-candle
    // cargo_test(&["-p", "burn-core", "--features", "test-candle"]);

    // Run cargo test --features test-wgpu
    cargo_test(&["-p", "burn-core", "--features", "test-wgpu"]);
}

// Test burn-dataset features
fn burn_dataset_features_std() {
    println!("\n\nRun checks for burn-dataset features");

    // Run cargo build --all-features
    cargo_build(&["-p", "burn-dataset", "--all-features"]);

    // Run cargo test --all-features
    cargo_test(&["-p", "burn-dataset", "--all-features"]);

    // Run cargo doc --all-features
    cargo_doc(&["-p", "burn-dataset", "--all-features"]);
}

fn std_checks() {
    // Set RUSTDOCFLAGS environment variable to treat warnings as errors
    // for the documentation build
    env::set_var("RUSTDOCFLAGS", "-D warnings");

    println!("Running std checks");

    // Build each workspace
    cargo_build(&["--workspace"]);

    // Test each workspace
    cargo_test(&["--workspace"]);

    // Check format
    cargo_fmt();

    // Check clippy lints
    cargo_clippy();

    // Produce documentation for each workspace
    cargo_doc(&["--workspace"]);

    // Test burn-dataset features
    burn_dataset_features_std();

    // Test burn-core with tch and wgpu backend
    burn_core_std();
}

fn check_typos() {
    // Install typos-cli
    cargo_install(&["typos-cli", "--version", "1.16.5"]);

    println!("Running typos check \n\n");

    // Run typos command as child process
    let typos = Command::new("typos")
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()) // Send stderr directly to terminal
        .spawn()
        .expect("Failed to run typos");

    // Handle typos child process
    handle_child_process(typos, "Failed to wait for typos child process");
}

fn main() {
    // Start time measurement
    let start = Instant::now();

    // The environment can assume ONLY "std", "no_std", "typos" as values.
    //
    // Depending on the input argument, the respective environment checks
    // are run.
    //
    // If no environment has been passed, run all checks.
    match env::args()
        .nth(
            1, /* Index of the first argument, because 0 is the binary name */
        )
        .as_deref()
    {
        Some("std") => std_checks(),
        Some("no_std") => no_std_checks(),
        Some("typos") => check_typos(),
        Some(_) | None => {
            /* Run all checks */
            check_typos();
            std_checks();
            no_std_checks();
        }
    }

    // Stop time measurement
    //
    // Compute runtime duration
    let duration = start.elapsed();

    // Print duration
    println!("Time elapsed for the current execution: {:?}", duration);
}
