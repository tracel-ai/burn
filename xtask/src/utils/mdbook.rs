use std::{
    collections::HashMap,
    path::Path,
    process::{Command, Stdio},
};

use crate::utils::process::handle_child_process;

use super::Params;

/// Run an mdbook command with the passed directory as the current directory
pub(crate) fn run_mdbook_with_path<P: AsRef<Path>>(
    command: &str,
    params: Params,
    envs: HashMap<&str, String>,
    path: Option<P>,
    error: &str,
) {
    info!("mdbook {} {}\n", command, params.params.join(" "));
    let mut mdbook = Command::new("mdbook");
    mdbook
        .envs(&envs)
        .arg(command)
        .args(&params.params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()); // Send stderr directly to terminal

    if let Some(path) = path {
        mdbook.current_dir(path);
    }

    // Handle mdbook child process
    let mdbook_process = mdbook.spawn().expect(error);
    handle_child_process(mdbook_process, "mdbook process should run flawlessly");
}
