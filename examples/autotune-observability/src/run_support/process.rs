use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::thread;

use super::RunMsg;

/// Run `cargo <args>` in the example dir, streaming interleaved stdout+stderr as [`RunMsg::Line`]
/// and finishing with [`RunMsg::Done`].
pub(crate) fn stream_command(args: Vec<String>, tx: Sender<RunMsg>) {
    let mut child = match Command::new("cargo")
        .current_dir(crate::example_dir())
        .env("CARGO_TERM_COLOR", "always")
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

    let ok = child.wait().map(|s| s.success()).unwrap_or(false);
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
