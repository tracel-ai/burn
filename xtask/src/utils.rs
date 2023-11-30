use serde_json::Value;
use std::{process::Command, time::Duration};

pub(crate) enum WorkspaceMemberType {
    Crate,
    Example,
}

#[derive(Debug)]
pub(crate) struct WorkspaceMember {
    pub name: String,
    pub path: String,
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

            let workspace_path = workspace_path.replace("(path+file://", "").replace(')', "");

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

/// Start log group
pub(crate) fn start_log_group(title: String) {
    // When running on CI, uses special grouping log
    if std::env::var("CI").is_ok() {
        println!("::group::{}", title);
    } else {
        println!("\n\n{}", title);
    }
}

/// End log group
pub(crate) fn end_log_group() {
    // When running on CI, uses special grouping log
    if std::env::var("CI").is_ok() {
        println!("::endgroup::");
    }
}

/// Print duration as HH:MM:SS format
pub(crate) fn pretty_print_duration(duration: &Duration) -> String {
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
