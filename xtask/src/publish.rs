//! This script publishes a crate on `crates.io`.
//!
//! To run the script:
//!
//! cargo xtask publish INPUT_CRATE

use std::{collections::HashMap, env, process::Command, str};

use crate::{
    endgroup, group,
    logging::init_logger,
    utils::{cargo::run_cargo, Params},
};

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

fn publish(crate_name: String) {
    // Perform dry-run to ensure everything is good for publishing
    let dry_run_params = Params::from(["-p", &crate_name, "--dry-run"]);

    run_cargo(
        "publish",
        dry_run_params,
        HashMap::new(),
        "The cargo publish --dry-run should complete successfully, indicating readiness for actual publication",
    );

    let crates_io_token =
        env::var(CRATES_IO_API_TOKEN).expect("Failed to retrieve the crates.io API token");
    let envs = HashMap::from([("CRATES_IO_API_TOKEN", crates_io_token.clone())]);
    let publish_params = Params::from(vec!["-p", &crate_name, "--token", &crates_io_token]);

    // Actually publish the crate
    run_cargo(
        "publish",
        publish_params,
        envs,
        "The crate should be successfully published",
    );
}

pub(crate) fn run(crate_name: String) -> anyhow::Result<()> {
    // Setup logger
    init_logger().init();

    group!("Publishing {}...\n", crate_name);

    // Retrieve local version for crate
    let local_version = local_version(&crate_name);
    info!("{crate_name} local version: {local_version}");

    // Retrieve remote version for crate if it exists
    match remote_version(&crate_name) {
        Some(remote_version) => {
            info!("{crate_name} remote version: {remote_version}\n");

            // Early return if we don't need to publish the crate
            if local_version == remote_version {
                info!("Remote version {remote_version} is up to date, skipping deployment");
                return Ok(());
            }
        }
        None => info!("\nFirst time publishing {crate_name} on crates.io!\n"),
    }

    // Publish the crate
    publish(crate_name);

    endgroup!();

    Ok(())
}
