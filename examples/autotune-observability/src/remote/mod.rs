//! Running the workload on a remote host over SSH while the UI stays local.
//!
//! The flow mirrors the local one but across the network: connect, push the locally-patched repos
//! (burn plus any cubecl/cubek pointed to by an active `[patch]`), `cargo run` the runner there,
//! then pull the produced `autotune.log` back into the local `runs/<id>/` so the existing parser
//! and UI work unchanged. Progress is streamed over the same [`RunMsg`] channel the local runner
//! uses, so the UI's poll loop is oblivious to which transport produced it.

mod config;
mod layout;
mod session;
mod ssh_config;
mod sync;

pub(crate) use config::RemoteConfig;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use session::{ExecOutcome, Remote};
use crate::run_support::{ProblemKind, RunMsg};

/// Everything a remote run needs, assembled on the UI thread and moved to the worker.
pub(crate) struct RemoteRun {
    pub feature: String,
    pub backend: String,
    pub problem: ProblemKind,
    pub input: String,
    pub output: String,
    pub shapes: Vec<Vec<usize>>,
    pub id: String,
    pub local_run_dir: PathBuf,
    /// Bypass the sync cache and re-check every file against the remote.
    pub force_sync: bool,
    /// Pass `--no-throughput-cache` to the runner so the peak bound is re-benchmarked.
    pub disable_throughput_cache: bool,
    /// Disable short circuits in autotune.
    pub disable_short_circuit: bool,
}

/// Drive a full remote run and stream progress, always ending with [`RunMsg::Done`].
pub(crate) fn run_remote(
    cfg: RemoteConfig,
    run: RemoteRun,
    cancel: Arc<AtomicBool>,
    tx: Sender<RunMsg>,
) {
    let ok = match execute(&cfg, &run, &cancel, &tx) {
        Ok(ok) => ok,
        Err(err) => {
            let _ = tx.send(RunMsg::Line(format!("remote error: {err}")));
            false
        }
    };
    let _ = tx.send(RunMsg::Done { ok });
}

/// Report cancellation and return `true` when the cancel flag is set.
fn stop_if_canceled(cancel: &AtomicBool, tx: &Sender<RunMsg>) -> bool {
    if cancel.load(Ordering::Relaxed) {
        let _ = tx.send(RunMsg::Line("[Canceled by user]".to_string()));
        true
    } else {
        false
    }
}

/// Connect, detect the remote OS, and return a one-line summary — backing the "Test connection"
/// button without touching any files.
pub(crate) fn test_connection(cfg: &RemoteConfig) -> Result<String, String> {
    let remote = Remote::connect(&cfg.host, &cfg.password)?;
    Ok(format!(
        "Connected to {} — {} host, base {}",
        cfg.host,
        remote.platform.label(),
        base_dir(&remote, cfg)
    ))
}

fn execute(
    cfg: &RemoteConfig,
    run: &RemoteRun,
    cancel: &AtomicBool,
    tx: &Sender<RunMsg>,
) -> Result<bool, String> {
    let _ = tx.send(RunMsg::Line(format!("connecting to {}…", cfg.host)));
    let remote = Remote::connect(&cfg.host, &cfg.password)?;
    let base = base_dir(&remote, cfg);
    let _ = tx.send(RunMsg::Line(format!(
        "{} host, base {base}",
        remote.platform.label()
    )));

    for repo in layout::repos_to_sync() {
        if stop_if_canceled(cancel, tx) {
            return Ok(false);
        }
        let remote_root = format!("{base}/{}", repo.name);
        let _ = tx.send(RunMsg::Line(format!(
            "syncing {} → {remote_root}",
            repo.local.display()
        )));
        remote.ensure_dir(&remote_root)?;
        let completed = sync::sync_tree(
            &remote,
            &repo.local,
            &remote_root,
            &cfg.host,
            run.force_sync,
            cancel,
            tx,
        )?;
        if !completed {
            let _ = tx.send(RunMsg::Line("[Canceled by user]".to_string()));
            return Ok(false);
        }
    }

    let mut remote_example = format!("{base}/{}", layout::root_name());
    let rel = layout::example_rel();
    if !rel.as_os_str().is_empty() {
        remote_example.push('/');
        remote_example.push_str(&rel.to_string_lossy().replace('\\', "/"));
    }
    let remote_run_dir = format!("{remote_example}/runs/{}", run.id);
    remote.ensure_dir(&remote_run_dir)?;

    if stop_if_canceled(cancel, tx) {
        return Ok(false);
    }
    let command = remote.run_command(&remote_example, &cargo_tail(run, &remote_run_dir));
    let _ = tx.send(RunMsg::Line(format!("$ {command}")));
    match remote.exec_stream(&command, cancel, tx)? {
        ExecOutcome::Canceled => {
            remote.kill_matching(&run.id);
            let _ = tx.send(RunMsg::Line("[Canceled by user]".to_string()));
            return Ok(false);
        }
        ExecOutcome::Exited(status) if status != 0 => {
            let _ = tx.send(RunMsg::Line(format!("runner exited with status {status}")));
            return Ok(false);
        }
        ExecOutcome::Exited(_) => {}
    }

    std::fs::create_dir_all(&run.local_run_dir).map_err(|e| e.to_string())?;
    remote.download(
        &format!("{remote_run_dir}/autotune.log"),
        &run.local_run_dir.join("autotune.log"),
    )?;
    // meta.txt is a convenience; don't fail the run if it's absent.
    let _ = remote.download(
        &format!("{remote_run_dir}/meta.txt"),
        &run.local_run_dir.join("meta.txt"),
    );
    let _ = tx.send(RunMsg::Line("fetched autotune.log".to_string()));
    Ok(true)
}

/// The remote base directory: the user's value (with a leading `~` expanded on unix), or the
/// remote's own temp dir when the field is left blank.
fn base_dir(remote: &Remote, cfg: &RemoteConfig) -> String {
    let configured = cfg.base_dir.trim();
    if configured.is_empty() {
        return format!("{}/burn-remote", remote.temp_dir);
    }
    expand_tilde(&configured.replace('\\', "/"), &remote.home)
}

/// Expand a leading `~` against the remote home; other paths are returned unchanged.
fn expand_tilde(path: &str, home: &str) -> String {
    if path == "~" {
        home.to_string()
    } else if let Some(rest) = path.strip_prefix("~/") {
        format!("{home}/{rest}")
    } else {
        path.to_string()
    }
}

/// The OS-independent `cargo run …` command. Always release, since debug autotune timings are
/// meaningless.
fn cargo_tail(run: &RemoteRun, remote_run_dir: &str) -> String {
    let mut env_prefix = String::new();
    if run.disable_short_circuit {
        env_prefix.push_str("CUBECL_AUTOTUNE_SHORT_CIRCUIT=0 ");
    }
    let mut command = format!(
        "{env_prefix}cargo run --release --bin runner --features {} -- \
         --backend {} --problem {} --input {} --output {}",
        run.feature,
        run.backend,
        run.problem.name(),
        run.input,
        run.output,
    );
    for shape in &run.shapes {
        let shape_str = shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("x");
        command.push_str(&format!(" --shape {shape_str}"));
    }
    command.push_str(&format!(" --run-dir \"{remote_run_dir}\""));
    if run.disable_throughput_cache {
        command.push_str(" --no-throughput-cache");
    }
    command
}

#[cfg(test)]
mod tests {
    use super::expand_tilde;

    #[test]
    fn expands_leading_tilde_only() {
        assert_eq!(expand_tilde("~/burn-remote", "/home/me"), "/home/me/burn-remote");
        assert_eq!(expand_tilde("~", "/home/me"), "/home/me");
        assert_eq!(expand_tilde("/opt/burn", "/home/me"), "/opt/burn");
        assert_eq!(expand_tilde("a/~/b", "/home/me"), "a/~/b");
    }
}
