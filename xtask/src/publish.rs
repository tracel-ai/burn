//! This script publishes a crate on `crates.io`.
//!
//! To run the script:
//!
//! cargo xtask publish INPUT_CRATE

use std::env;
use std::process::{Command, Stdio};
use std::str;

use crate::{endgroup, group};

// Crates.io API token
const CRATES_IO_API_TOKEN: &str = "CRATES_IO_API_TOKEN";

// Obtain local crate version
fn local_version(crate_name: &str) -> String {
    // Obtain local crate version contained in cargo pkgid data
    let cargo_pkgid_output = Command::new("cargo")
        .args(["pkgid", "-p", crate_name])
        .output()
        .expect("Failed to run cargo pkgid");

    // Convert cargo pkgid output into a str
    let cargo_pkgid_str = str::from_utf8(&cargo_pkgid_output.stdout)
        .expect("Failed to convert pkgid output into a str");

    // Extract only the local crate version from str
    let (_, local_version) = cargo_pkgid_str
        .split_once('#')
        .expect("Failed to get local crate version");

    local_version.trim_end().to_string()
}

// Obtain remote crate version
fn remote_version(crate_name: &str) -> Option<String> {
    // Obtain remote crate version contained in cargo search data
    let cargo_search_output = Command::new("cargo")
        .args(["search", crate_name, "--limit", "1"])
        .output()
        .expect("Failed to run cargo search");

    // Cargo search returns an empty string in case of a crate not present on
    // crates.io
    if cargo_search_output.stdout.is_empty() {
        None
    } else {
        // Convert cargo search output into a str
        let remote_version_str = str::from_utf8(&cargo_search_output.stdout)
            .expect("Failed to convert cargo search output into a str");

        // Extract only the remote crate version from str
        remote_version_str
            .split_once('=')
            .and_then(|(_, second)| second.trim_start().split_once(' '))
            .map(|(s, _)| s.trim_matches('"').to_string())
    }
}

// Run cargo publish
fn cargo_publish(params: &[&str]) {
    // Run cargo publish
    let mut cargo_publish = Command::new("cargo")
        .arg("publish")
        .arg("--color=always")
        .args(params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()) // Send stderr directly to terminal
        .spawn()
        .expect("Failed to run cargo publish");

    // Wait for cargo publish command to finish
    let status = cargo_publish
        .wait()
        .expect("Failed to wait for cargo publish child process");

    // If exit status is not a success, terminate the process with an error
    if !status.success() {
        // Use the exit code associated to a command to terminate the process,
        // if any exit code had been found, use the default value 1
        std::process::exit(status.code().unwrap_or(1));
    }
}

// Publishes a crate
fn publish(crate_name: String) {
    // Run cargo publish --dry-run
    cargo_publish(&["-p", &crate_name, "--dry-run"]);

    let crates_io_token =
        env::var(CRATES_IO_API_TOKEN).expect("Failed to retrieve the crates.io API token");

    // Publish crate
    cargo_publish(&["-p", &crate_name, "--token", &crates_io_token]);
}

pub(crate) fn run(crate_name: String) -> anyhow::Result<()> {
    group!("Publishing {}...\n", crate_name);

    // Retrieve local version for crate
    let local_version = local_version(&crate_name);

    // Print local version for crate
    info!("{crate_name} local version: {local_version}");

    // Retrieve remote version for crate
    //
    // If remote version is None, the crate will be published for the first time
    // on crates.io
    if let Some(remote_version) = remote_version(&crate_name) {
        // Print local version for crate
        info!("{crate_name} remote version: {remote_version}\n");

        // If local and remote versions are equal, do not publish
        if local_version == remote_version {
            info!("Remote version {remote_version} is up to date, skipping deployment");
        } else {
            // Publish crate
            publish(crate_name);
        }
    } else {
        // Print crate publishing message
        info!("\nFirst time publishing {crate_name} on crates.io!\n");
        // Publish crate
        publish(crate_name);
    }

    endgroup!();

    Ok(())
}
