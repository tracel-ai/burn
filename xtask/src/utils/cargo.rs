use std::{
    collections::HashMap,
    path::Path,
    process::{Command, Stdio},
};

use crate::{endgroup, group, utils::process::handle_child_process};

use super::Params;

/// Run a cargo command
pub(crate) fn run_cargo(command: &str, params: Params, envs: HashMap<&str, String>, error: &str) {
    run_cargo_with_path::<String>(command, params, envs, None, error)
}

/// Run a cargo command with the passed directory as the current directory
pub(crate) fn run_cargo_with_path<P: AsRef<Path>>(
    command: &str,
    params: Params,
    envs: HashMap<&str, String>,
    path: Option<P>,
    error: &str,
) {
    info!("cargo {} {}\n", command, params.params.join(" "));
    let mut cargo = Command::new("cargo");
    cargo
        .env("CARGO_INCREMENTAL", "0")
        .envs(&envs)
        .arg(command)
        .args(&params.params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()); // Send stderr directly to terminal

    if let Some(path) = path {
        cargo.current_dir(path);
    }

    // Handle cargo child process
    let cargo_process = cargo.spawn().expect(error);
    handle_child_process(cargo_process, "Cargo process should run flawlessly");
}

/// Ensure that a cargo crate is installed
pub(crate) fn ensure_cargo_crate_is_installed(crate_name: &str) {
    if !is_cargo_crate_installed(crate_name) {
        group!("Cargo: install crate '{}'", crate_name);
        run_cargo(
            "install",
            [crate_name].into(),
            HashMap::new(),
            &format!("crate '{}' should be installed", crate_name),
        );
        endgroup!();
    }
}

/// Returns true if the passed cargo crate is installed locally
fn is_cargo_crate_installed(crate_name: &str) -> bool {
    let output = Command::new("cargo")
        .arg("install")
        .arg("--list")
        .output()
        .expect("Should get the list of installed cargo commands");
    let output_str = String::from_utf8_lossy(&output.stdout);
    output_str.lines().any(|line| line.contains(crate_name))
}
