use std::process::{Command, Stdio};

use crate::{endgroup, group, utils::process::handle_child_process};

use super::Params;

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
    let cargo_process = rustup.spawn().expect(expected);
    handle_child_process(cargo_process, "Failed to wait for rustup child process");
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
