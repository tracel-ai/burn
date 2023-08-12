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
use std::process::{Command, Output};
use std::str;
use std::time::Instant;

// Targets constants
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";

// Write stdout and stderr output on shell.
// If there exit status of a command is not a success, terminate the process
// with an error.
fn stdout_and_stderr_write(output: Output, message_stdout: &str, message_stderr: &str) {
    if !output.stdout.is_empty() {
        println!("{}", str::from_utf8(&output.stdout).expect(message_stdout));
    }

    if !output.stderr.is_empty() {
        println!("{}", str::from_utf8(&output.stderr).expect(message_stderr));
    }

    // If exit status is not a success, terminate the process with an error
    if !output.status.success() {
        // Use the exit code associated to a command to terminate the process,
        // if any exit code had been found, use the default value 1
        std::process::exit(output.status.code().unwrap_or(1));
    }
}

// Define and run rustup command
fn rustup(target: &str) {
    // Rustup arguments
    let args = ["target", "add", target];

    // Print rustup command
    println!("rustup {}\n\n", args.join(" "));

    // Run rustup command
    let rustup = Command::new("rustup")
        .args(args)
        .output()
        .expect("Failed to run rustup");

    // Write rustup output either on stdout or on stderr
    stdout_and_stderr_write(
        rustup,
        "Failed to write rustup output on stdout",
        "Failed to write rustup output on stderr",
    );
}

// Define and run a cargo command
fn run_cargo(
    command: &str,
    first_params: &[&str],
    second_params: &[&str],
    error: &str,
    stdout_error: &str,
    stderr_error: &str,
) {
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
        &["--color=always"],
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
        &["--color=always", "--", "--color=always"],
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
        &["--", "--color=always"],
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
        &["--color=always"],
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
        &["--color=always"],
        "Failed to run cargo doc",
        "Failed to write cargo doc output on stdout",
        "Failed to write cargo doc output on stderr",
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
    println!("\n\nChecks for no_std environment...\n\n");

    println!("Install Wasm32 target\n");
    // Install wasm32 target
    rustup(WASM32_TARGET);

    println!("\nInstall ARM target\n");
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
