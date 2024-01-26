use serde_json::Value;
use std::{
    collections::HashMap,
    path::Path,
    process::{Child, Command, Stdio},
    time::Duration,
};

use crate::{endgroup, group};

// Cargo utils -----------------------------------------------------------

/// Run a cargo command
pub(crate) fn run_cargo(command: &str, params: Params, envs: HashMap<String, String>, error: &str) {
    run_cargo_with_path::<String>(command, params, envs, None, error)
}

/// Run acargo command with the passed directory as the current directory
pub(crate) fn run_cargo_with_path<P: AsRef<Path>>(
    command: &str,
    params: Params,
    envs: HashMap<String, String>,
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
    let rustup_process = cargo.spawn().expect(error);
    handle_child_process(rustup_process, "Cargo process should run flawlessly");
}

/// Ensure that a cargo command is installed
pub(crate) fn ensure_cargo_command_is_installed(command: &str) {
    if !is_cargo_command_installed(command) {
        group!("Cargo: install {} command", command);
        run_cargo(
            "install",
            [command].into(),
            HashMap::new(),
            &format!("{} should be installed", command),
        );
        endgroup!();
    }
}

/// Returns true if the passed cargo command is installed locally
fn is_cargo_command_installed(command: &str) -> bool {
    let output = Command::new("cargo")
        .arg("install")
        .arg("--list")
        .output()
        .expect("Should get the list of installed cargo commands");
    let output_str = String::from_utf8_lossy(&output.stdout);
    output_str.lines().any(|line| line.contains(command))
}

pub(crate) struct Params {
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

impl From<Vec<&str>> for Params {
    fn from(value: Vec<&str>) -> Self {
        Self {
            params: value.iter().map(|s| s.to_string()).collect(),
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

// Rustup utils --------------------------------------------------------------

/// Run rustup command
pub(crate) fn rustup(command: &str, params: Params, expected: &str) {
    info!("rustup {} {}\n", command, params);
    // Run rustup
    let mut rustup = Command::new("rustup");
    rustup
        .arg(command)
        .args(params.params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()); // Send stderr directly to terminal
    let rustup_process = rustup.spawn().expect(expected);
    handle_child_process(rustup_process, "Failed to wait for rustup child process");
}

/// Add a Rust target
pub(crate) fn rustup_add_target(target: &str) {
    group!("Rustup: add target {}", target);
    rustup(
        "target",
        Params::from(["add", target]),
        "Target should be added",
    );
    endgroup!();
}

/// Add a Rust component
pub(crate) fn rustup_add_component(component: &str) {
    group!("Rustup: add component {}", component);
    rustup(
        "component",
        Params::from(["add", component]),
        "Component should be added",
    );
    endgroup!();
}

// Returns the output of the rustup command to get the installed targets
pub(crate) fn rustup_get_installed_targets() -> String {
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .stdout(Stdio::piped())
        .output()
        .expect("Rustup command should execute successfully");
    String::from_utf8(output.stdout).expect("Output should be valid UTF-8")
}

/// Returns true if the current toolchain is the nightly
pub(crate) fn is_current_toolchain_nightly() -> bool {
    let output = Command::new("rustup")
        .arg("show")
        .output()
        .expect("Should get the list of installed Rust toolchains");
    let output_str = String::from_utf8_lossy(&output.stdout);
    for line in output_str.lines() {
        // look for the "rustc.*-nightly" line
        if line.contains("rustc") && line.contains("-nightly") {
            return true;
        }
    }
    // assume we are using a stable toolchain if we did not find the nightly compiler
    false
}

// Workspace utils -----------------------------------------------------------

pub(crate) enum WorkspaceMemberType {
    Crate,
    Example,
}

#[derive(Debug)]
pub(crate) struct WorkspaceMember {
    pub(crate) name: String,
    pub(crate) path: String,
}

impl WorkspaceMember {
    fn new(name: String, path: String) -> Self {
        Self { name, path }
    }
}

/// Get project workspaces
pub(crate) fn get_workspaces(w_type: WorkspaceMemberType) -> Vec<WorkspaceMember> {
    // Run `cargo metadata` command to get project metadata
    let output = Command::new("cargo")
        .arg("metadata")
        .output()
        .expect("Failed to execute command");

    // Parse the JSON output
    let metadata: Value = serde_json::from_slice(&output.stdout).expect("Failed to parse JSON");

    // Extract workspaces from the metadata, excluding examples/ and xtask
    let workspaces = metadata["workspace_members"]
        .as_array()
        .expect("Expected an array of workspace members")
        .iter()
        .filter_map(|member| {
            let parts: Vec<_> = member.as_str()?.split_whitespace().collect();
            let (workspace_name, workspace_path) =
                (parts.first()?.to_owned(), parts.last()?.to_owned());

            let prefix = if cfg!(target_os = "windows") {
                "(path+file:///"
            } else {
                "(path+file://"
            };
            let workspace_path = workspace_path.replace(prefix, "").replace(')', "");

            match w_type {
                WorkspaceMemberType::Crate
                    if workspace_name != "xtask" && !workspace_path.contains("examples/") =>
                {
                    Some(WorkspaceMember::new(
                        workspace_name.to_string(),
                        workspace_path.to_string(),
                    ))
                }
                WorkspaceMemberType::Example
                    if workspace_name != "xtask" && workspace_path.contains("examples/") =>
                {
                    Some(WorkspaceMember::new(
                        workspace_name.to_string(),
                        workspace_path.to_string(),
                    ))
                }
                _ => None,
            }
        })
        .collect();

    workspaces
}

// Various utils -------------------------------------------------------------

/// Print duration as HH:MM:SS format
pub(crate) fn format_duration(duration: &Duration) -> String {
    let seconds = duration.as_secs();
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let remaining_minutes = minutes % 60;
    let remaining_seconds = seconds % 60;

    format!(
        "{:02}:{:02}:{:02}",
        hours, remaining_minutes, remaining_seconds
    )
}

// Handle child process
pub(crate) fn handle_child_process(mut child: Child, error: &str) {
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
pub(crate) fn run_command(command: &str, args: &[&str], command_error: &str, child_error: &str) {
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
