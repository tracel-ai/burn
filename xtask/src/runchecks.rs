//! This script is run before a PR is created.
//!
//! It is used to check that the code compiles and passes all tests.
//!
//! It is also used to check that the code is formatted correctly and passes clippy.
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
fn run_cargo(command: &str, params: Params, error: &str) {
    // Print cargo command
    println!("\ncargo {} {}\n", command, params);

    // Run cargo
    let cargo = Command::new("cargo")
        .arg(command)
        .args(params.params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()) // Send stderr directly to terminal
        .spawn()
        .expect(error);

    // Handle cargo child process
    handle_child_process(cargo, "Failed to wait for cargo child process");
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
    // Run cargo fmt
    run_cargo(
        "fmt",
        ["--check", "--all", "--", "--color=always"].into(),
        "Failed to run cargo fmt",
    );
}

// Run cargo clippy command
fn cargo_clippy() {
    if std::env::var("RUST_MATRIX").map_or(false, |matrix| matrix != "stable") {
        return;
    }
    // Run cargo clippy
    run_cargo(
        "clippy",
        ["--color=always", "--", "-D", "warnings"].into(),
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
    println!("\nRun checks for `{}` crate", crate_name);

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
}

// Run no_std checks
fn no_std_checks() {
    println!("Checks for no_std environment...\n\n");

    // Install wasm32 target
    rustup(WASM32_TARGET);

    // Install ARM target
    rustup(ARM_TARGET);

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
    println!("\n\nRun checks for burn-core crate with tch and wgpu backend");

    // Run cargo test --features test-tch
    cargo_test(["-p", "burn-core", "--features", "test-tch"].into());

    // Run cargo test --features test-wgpu
    cargo_test(["-p", "burn-core", "--features", "test-wgpu"].into());
}

// Test burn-dataset features
fn burn_dataset_features_std() {
    println!("\n\nRun checks for burn-dataset features");

    // Run cargo build --all-features
    cargo_build(["-p", "burn-dataset", "--all-features"].into());

    // Run cargo test --all-features
    cargo_test(["-p", "burn-dataset", "--all-features"].into());

    // Run cargo doc --all-features
    cargo_doc(["-p", "burn-dataset", "--all-features"].into());
}

fn std_checks() {
    // Set RUSTDOCFLAGS environment variable to treat warnings as errors
    // for the documentation build
    env::set_var("RUSTDOCFLAGS", "-D warnings");

    println!("Running std checks");

    // Build each workspace
    cargo_build(["--workspace", "--exclude=xtask"].into());

    // Test each workspace
    cargo_test(["--workspace"].into());

    // Check format
    cargo_fmt();

    // Check clippy lints
    cargo_clippy();

    // Produce documentation for each workspace
    cargo_doc(["--workspace"].into());

    // Test burn-dataset features
    burn_dataset_features_std();

    // Test burn-core with tch and wgpu backend
    burn_core_std();
}

fn check_typos() {
    // Install typos-cli
    cargo_install(["typos-cli", "--version", "1.16.5"].into());

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

fn check_examples() {
    println!("Checking examples compile \n\n");

    std::fs::read_dir("examples").unwrap().for_each(|dir| {
        let dir = dir.unwrap();
        let path = dir.path();
        // Skip if not a directory
        if !path.is_dir() {
            return;
        }
        if path.file_name().unwrap().to_str().unwrap() == "notebook" {
            // not a crate
            return;
        }
        let path = path.to_str().unwrap();
        println!("Checking {path} \n\n");

        let child = Command::new("cargo")
            .arg("check")
            .arg("--examples")
            .current_dir(dir.path())
            .stdout(Stdio::inherit()) // Send stdout directly to terminal
            .stderr(Stdio::inherit()) // Send stderr directly to terminal
            .spawn()
            .expect("Failed to check examples");

        // Handle typos child process
        handle_child_process(child, "Failed to wait for examples child process");
    });
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
    println!("Time elapsed for the current execution: {:?}", duration);

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
