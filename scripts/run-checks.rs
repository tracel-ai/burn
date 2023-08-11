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
//!     - `all` to perform checks using both `libstd` and `libcore`

use std::env;
use std::io::{self, Write};
use std::process::{Command, Output};
use std::str;
use std::time::Instant;

// If stdout is not empty, write it, otherwise
// write stderr and exit with code error 1
fn stdout_and_stderr_write(output: Output, message_stdout: &str, message_stderr: &str) {
    if !output.stdout.is_empty() {
        io::stdout()
            .write_all(&output.stdout)
            .expect(message_stdout);
    }

    if !output.stderr.is_empty() {
        io::stderr()
            .write_all(&output.stderr)
            .expect(message_stderr);
        std::process::exit(1);
    }
}

// Run rustup command
fn rustup(target: &str) {
    // Run rustup
    let rustup = Command::new("rustup")
        .args(["target", "add", target])
        .output()
        .expect("Failed to run rustup");

    // Write rustup output either on stdout or on stderr
    stdout_and_stderr_write(
        rustup,
        "Failed to write rustup output on stdout",
        "Failed to write rustup output on stderr",
    );
}

// Run a cargo command
fn run_cargo(command: &str, params: &[&str], error: &str, stdout_error: &str, stderr_error: &str) {
    // Run cargo
    let cargo = Command::new("cargo")
        .arg(command)
        .args(params)
        .output()
        .expect(error);

    // Write cargo output either on stdout or on stderr
    stdout_and_stderr_write(cargo, stdout_error, stderr_error);
}

// Run cargo build command
fn cargo_build(params: &[&str]) {
    // Run cargo build
    run_cargo(
        "build",
        params,
        "Failed to run cargo build",
        "Failed to write cargo build output on stdout",
        "Failed to write cargo build output on stderr",
    );
}

// Run cargo test command
fn cargo_test(params: &[&str]) {
    // Run cargo test
    run_cargo(
        "test",
        params,
        "Failed to run cargo test",
        "Failed to write cargo test output on stdout",
        "Failed to write cargo test output on stderr",
    );
}

// Run cargo fmt command
fn cargo_fmt() {
    // Run cargo fmt
    run_cargo(
        "fmt",
        &["--check", "--all"],
        "Failed to run cargo fmt",
        "Failed to write cargo fmt output on stdout",
        "Failed to write cargo fmt output on stderr",
    );
}

// Run cargo clippy command
fn cargo_clippy() {
    // Run cargo clippy
    run_cargo(
        "clippy",
        &["--", "-D", "warnings"],
        "Failed to run cargo clippy",
        "Failed to write cargo clippy output on stdout",
        "Failed to write cargo clippy output on stderr",
    );
}

// Run cargo doc command
fn cargo_doc(params: &[&str]) {
    // Run cargo doc
    run_cargo(
        "doc",
        params,
        "Failed to run cargo doc",
        "Failed to write cargo doc output on stdout",
        "Failed to write cargo doc output on stderr",
    );
}

// Build and test a crate in a no_std environment
fn build_and_test_no_std(crate_name: &str) {
    println!("\nRun checks for `{}` crate", crate_name);

    println!("\nBuild without defaults");
    // Run cargo build --no-default-features
    cargo_build(&["-p", crate_name, "--no-default-features"]);

    println!("\nTest without defaults");
    // Run cargo test --no-default-features
    cargo_test(&["-p", crate_name, "--no-default-features"]);

    println!("\nBuild for WebAssembly");
    // Run cargo build --no-default-features --target wasm32-unknown-unknowns
    cargo_build(&[
        "-p",
        crate_name,
        "--no-default-features",
        "--target",
        "wasm32-unknown-unknowns",
    ]);

    println!("\nBuild for ARM");
    // Run cargo build --no-default-features --target thumbv7m-none-eabi
    cargo_build(&[
        "-p",
        crate_name,
        "--no-default-features",
        "--target",
        "thumbv7m-none-eabi",
    ]);
}

// Run no_std checks
fn no_std_checks() {
    println!("Checks for no_std environment...\n\n");

    println!("Install Wasm32 target\n");
    // Install wasm32 target
    rustup("wasm32-unknown-unknown");

    println!("\nInstall ARM target\n");
    // Install ARM target
    rustup("thumbv7m-none-eabi");

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

    println!("\nTest with tch backend");
    // Run cargo test --features test-tch
    cargo_test(&["-p", "burn-core", "--features", "test-tch"]);

    println!("\nTest with wgpu backend");
    // Run cargo test --features test-wgpu
    cargo_test(&["-p", "burn-core", "--features", "test-wgpu"]);
}

// Test burn-dataset features
fn burn_dataset_features_std() {
    println!("\n\nRun checks for burn-dataset features");

    println!("\nBuild with all features");
    // Run cargo build --all-features
    cargo_build(&["-p", "burn-dataset", "--all-features"]);

    println!("\nTest with all features");
    // Run cargo test --all-features
    cargo_test(&["-p", "burn-dataset", "--all-features"]);

    println!("\nCheck documentation with all features");
    // Run cargo doc --all-features
    cargo_doc(&["-p", "burn-dataset", "--all-features"]);
}

fn std_checks() {
    // Set RUSTDOCFLAGS environment variable to treat warnings as errors
    // for the documentation build
    env::set_var("RUSTDOCFLAGS", "-D warnings");

    println!("\n\nRunning std checks");

    println!("\nBuild each workspace");
    // Build each workspace
    cargo_build(&["--workspace"]);

    println!("\nTest each workspace");
    // Test each workspace
    cargo_test(&["--workspace"]);

    println!("\nCheck format");
    // Check format
    cargo_fmt();

    println!("\nCheck clippy lints");
    // Check clippy lints
    cargo_clippy();

    println!("\nProduce documentation for each workspace");
    // Produce documentation for each workspace
    cargo_doc(&["--workspace"]);

    // Test burn-dataset features
    burn_dataset_features_std();

    // Test burn-core with tch and wgpu backend
    burn_core_std();
}

fn main() {
    // Get the environment
    let environment = env::args()
        .nth(1) // Index of the first argument, because 0 is the binary name
        .expect("You need to pass the environment as first argument!");

    // Start time measurement
    let start = Instant::now();

    // The environment can assume ONLY "all", "std" and "no_std" as values.
    //
    // Depending on the input argument, the respective environment checks
    // are run.
    //
    // If a wrong argument is passed, the program panics.
    match environment.as_str() {
        "all" => {
            std_checks();
            no_std_checks();
        }
        "std" => std_checks(),
        "no_std" => no_std_checks(),
        _ => {
            panic!("You can pass only 'all', 'std' and 'no_std' as values for the first argument!")
        }
    }

    // Stop time measurement
    //
    // Compute runtime duration
    let duration = start.elapsed();

    // Print duration
    println!("Time elapsed for the current execution: {:?}", duration);
}
