use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::thread;

use super::RunMsg;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Run `cargo <args>` in the example dir, streaming interleaved stdout+stderr as [`RunMsg::Line`]
/// and finishing with [`RunMsg::Done`].
pub(crate) fn stream_command(
    args: Vec<String>,
    envs: Vec<(String, String)>,
    tx: Sender<RunMsg>,
    cancel_flag: Arc<AtomicBool>,
) {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(crate::example_dir());
    cmd.env("CARGO_TERM_COLOR", "always");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let mut child = match cmd
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            let _ = tx.send(RunMsg::Line(format!("failed to launch cargo: {err}")));
            let _ = tx.send(RunMsg::Done { ok: false });
            return;
        }
    };

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let tx_out = tx.clone();
    let tx_err = tx.clone();
    let out_reader = thread::spawn(move || pipe_lines(stdout, &tx_out));
    let err_reader = thread::spawn(move || pipe_lines(stderr, &tx_err));

    let ok = loop {
        if cancel_flag.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = tx.send(RunMsg::Line(String::from("\n[Canceled by user]")));
            break false;
        }
        match child.try_wait() {
            Ok(Some(status)) => break status.success(),
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(_) => break false,
        }
    };
    let _ = out_reader.join();
    let _ = err_reader.join();
    let _ = tx.send(RunMsg::Done { ok });
}

fn pipe_lines(reader: impl std::io::Read, tx: &Sender<RunMsg>) {
    for line in BufReader::new(reader).lines().map_while(Result::ok) {
        if tx.send(RunMsg::Line(line)).is_err() {
            break;
        }
    }
}
