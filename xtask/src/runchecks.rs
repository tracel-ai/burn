//! This script is run before a PR is created.
//!
//! It is used to check that the code compiles and passes all tests.
//!
//! It is also used to check that the code is formatted correctly and passes clippy.

use std::{
    collections::HashMap,
    env,
    path::Path,
    process::{Command, Stdio},
    str,
};

use xtask_common::{
    anyhow, clap, endgroup, group,
    utils::{
        cargo::{run_cargo, run_cargo_with_path},
        process::{handle_child_process, run_command},
        rustup::{rustup_add_component, rustup_add_target},
        workspace::{get_workspace_members, WorkspaceMemberType},
        Params,
    },
};

// Targets constants
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";

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

impl CheckType {
    pub(crate) fn run(&self) -> anyhow::Result<()> {
        // The environment can assume ONLY "std", "no_std", "typos", "examples"
        //
        // Depending on the input argument, the respective environment checks
        // are run.
        //
        // If no environment has been passed, run all checks.
        match self {
            Self::Std => std_checks(),
            Self::NoStd => no_std_checks(),
            Self::Typos => check_typos(),
            Self::Examples => check_examples(),
            Self::All => {
                /* Run all checks */
                check_typos()?;
                std_checks()?;
                no_std_checks()?;
                check_examples()
            }
        }
    }
}

/// Run cargo build command
fn cargo_build(params: Params) -> anyhow::Result<()> {
    // Run cargo build
    run_cargo(
        "build",
        params + "--color=always",
        HashMap::new(),
        "Failed to run cargo build",
    )
}

/// Run cargo install command
fn cargo_install(params: Params) -> anyhow::Result<()> {
    // Run cargo install
    run_cargo(
        "install",
        params + "--color=always",
        HashMap::new(),
        "Failed to run cargo install",
    )
}

/// Run cargo test command
fn cargo_test(params: Params) -> anyhow::Result<()> {
    // Run cargo test
    run_cargo(
        "test",
        params + "--color=always" + "--" + "--color=always",
        HashMap::new(),
        "Failed to run cargo test",
    )
}

/// Run cargo fmt command
fn cargo_fmt() -> anyhow::Result<()> {
    group!("Cargo: fmt");
    run_cargo(
        "fmt",
        ["--check", "--all", "--", "--color=always"].into(),
        HashMap::new(),
        "Failed to run cargo fmt",
    )?;
    endgroup!();
    Ok(())
}

/// Run cargo clippy command
fn cargo_clippy() -> anyhow::Result<()> {
    if std::env::var("CI").is_ok() {
        return Ok(());
    }
    // Run cargo clippy
    run_cargo(
        "clippy",
        ["--color=always", "--all-targets", "--", "-D", "warnings"].into(),
        HashMap::new(),
        "Failed to run cargo clippy",
    )
}

/// Run cargo doc command
fn cargo_doc(params: Params) -> anyhow::Result<()> {
    // Run cargo doc
    run_cargo(
        "doc",
        params + "--color=always",
        HashMap::new(),
        "Failed to run cargo doc",
    )
}

// Build and test a crate in a no_std environment
fn build_and_test_no_std<const N: usize>(
    crate_name: &str,
    extra_args: [&str; N],
) -> anyhow::Result<()> {
    group!("Checks: {} (no-std)", crate_name);

    // Run cargo build --no-default-features
    cargo_build(Params::from(["-p", crate_name, "--no-default-features"]) + extra_args)?;

    // Run cargo test --no-default-features
    cargo_test(Params::from(["-p", crate_name, "--no-default-features"]) + extra_args)?;

    // Run cargo build --no-default-features --target wasm32-unknown-unknowns
    cargo_build(
        Params::from([
            "-p",
            crate_name,
            "--no-default-features",
            "--target",
            WASM32_TARGET,
        ]) + extra_args,
    )?;

    // Run cargo build --no-default-features --target thumbv7m-none-eabi
    cargo_build(
        Params::from([
            "-p",
            crate_name,
            "--no-default-features",
            "--target",
            ARM_TARGET,
        ]) + extra_args,
    )?;

    endgroup!();
    Ok(())
}

// Setup code coverage
fn setup_coverage() {
    // Install llvm-tools-preview
    rustup_add_component("llvm-tools-preview").expect("rustup component should be installed");

    // Set coverage environment variables
    env::set_var("RUSTFLAGS", "-Cinstrument-coverage");
    env::set_var("LLVM_PROFILE_FILE", "burn-%p-%m.profraw");
}

// Run grcov to produce lcov.info
fn run_grcov() -> anyhow::Result<()> {
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
    )
}

// Run no_std checks
fn no_std_checks() -> anyhow::Result<()> {
    // Install wasm32 target
    rustup_add_target(WASM32_TARGET)?;

    // Install ARM target
    rustup_add_target(ARM_TARGET)?;

    // Run checks for the following crates
    build_and_test_no_std("burn", [])?;
    build_and_test_no_std("burn-core", [])?;
    build_and_test_no_std("burn-common", [])?;
    build_and_test_no_std("burn-tensor", [])?;
    build_and_test_no_std("burn-ndarray", [])?;
    build_and_test_no_std("burn-no-std-tests", [])?;
    Ok(())
}

// Test burn-core with tch and wgpu backend
fn burn_core_std() -> anyhow::Result<()> {
    // Run cargo test --features test-tch, record-item-custom-serde
    group!("Test: burn-core (tch) and record-item-custom-serde");
    cargo_test(
        [
            "-p",
            "burn-core",
            "--features",
            "test-tch,record-item-custom-serde,",
        ]
        .into(),
    )?;
    endgroup!();

    // Run cargo test --features test-wgpu
    if std::env::var("DISABLE_WGPU").is_err() {
        group!("Test: burn-core (wgpu)");
        cargo_test(["-p", "burn-core", "--features", "test-wgpu"].into())?;
        endgroup!();
    }
    Ok(())
}

// Test burn-dataset features
fn burn_dataset_features_std() -> anyhow::Result<()> {
    group!("Checks: burn-dataset (all-features)");

    // Run cargo build --all-features
    cargo_build(["-p", "burn-dataset", "--all-features"].into())?;

    // Run cargo test --all-features
    cargo_test(["-p", "burn-dataset", "--all-features"].into())?;

    // Run cargo doc --all-features
    cargo_doc(["-p", "burn-dataset", "--all-features", "--no-deps"].into())?;

    endgroup!();
    Ok(())
}

// macOS only checks
#[cfg(target_os = "macos")]
fn macos_checks() -> anyhow::Result<()> {
    // Leverages the macOS Accelerate framework: https://developer.apple.com/documentation/accelerate
    group!("Checks: burn-candle (accelerate)");
    cargo_test(["-p", "burn-candle", "--features", "accelerate"].into())?;
    endgroup!();

    // Leverages the macOS Accelerate framework: https://developer.apple.com/documentation/accelerate
    group!("Checks: burn-ndarray (accelerate)");
    cargo_test(["-p", "burn-ndarray", "--features", "blas-accelerate"].into())?;
    endgroup!();
    Ok(())
}

fn std_checks() -> anyhow::Result<()> {
    // Set RUSTDOCFLAGS environment variable to treat warnings as errors
    // for the documentation build
    env::set_var("RUSTDOCFLAGS", "-D warnings");

    // Check if COVERAGE environment variable is set
    let is_coverage = std::env::var("COVERAGE").is_ok();
    let disable_wgpu = std::env::var("DISABLE_WGPU").is_ok();

    // Check format
    cargo_fmt()?;

    // Check clippy lints
    cargo_clippy()?;

    // Produce documentation for each workspace member
    group!("Docs: crates");
    let mut params = Params::from(["--workspace", "--no-deps"]);
    // Exclude burn-cuda on all platforms
    params.params.push("--exclude".to_string());
    params.params.push("burn-cuda".to_string());
    cargo_doc(params)?;
    endgroup!();

    // Setup code coverage
    if is_coverage {
        setup_coverage();
    }

    // Build & test each member in workspace
    let members = get_workspace_members(WorkspaceMemberType::Crate);
    for member in members {
        if disable_wgpu && member.name == "burn-wgpu" {
            continue;
        }
        if member.name == "burn-cuda" {
            // burn-cuda requires CUDA Toolkit which is not currently setup on our CI runners
            continue;
        }
        if member.name == "burn-tch" {
            continue;
        }

        group!("Checks: {}", member.name);
        cargo_build(Params::from(["-p", &member.name]))?;
        cargo_test(Params::from(["-p", &member.name]))?;
        endgroup!();
    }

    // Test burn-candle with accelerate (macOS only)
    #[cfg(target_os = "macos")]
    macos_checks()?;

    // Test burn-dataset features
    burn_dataset_features_std()?;

    // Test burn-core with tch and wgpu backend
    burn_core_std()?;

    // Run grcov and produce lcov.info
    if is_coverage {
        run_grcov()?;
    }
    Ok(())
}

fn check_typos() -> anyhow::Result<()> {
    // This path defines where typos-cli is installed on different
    // operating systems.
    let typos_cli_path = std::env::var("CARGO_HOME")
        .map(|v| std::path::Path::new(&v).join("bin/typos-cli"))
        .unwrap();

    // Do not run cargo install on CI to speed up the computation.
    // Check whether the file has been installed on
    if std::env::var("CI").is_err() && !typos_cli_path.exists() {
        // Install typos-cli
        cargo_install(["typos-cli", "--version", "1.16.5"].into())?;
    }

    info!("Running typos check \n\n");

    // Run typos command as child process
    let typos = Command::new("typos")
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()) // Send stderr directly to terminal
        .spawn()
        .expect("Failed to run typos");

    // Handle typos child process
    handle_child_process(typos, "Failed to wait for typos child process")
}

fn check_examples() -> anyhow::Result<()> {
    let members = get_workspace_members(WorkspaceMemberType::Example);
    for member in members {
        if member.name == "notebook" {
            continue;
        }

        group!("Checks: Example - {}", member.name);
        let path = Path::new(&member.path);
        run_cargo_with_path(
            "check",
            ["--examples"].into(),
            HashMap::new(),
            path,
            "Failed to check example",
        )?;
        endgroup!();
    }
    Ok(())
}
