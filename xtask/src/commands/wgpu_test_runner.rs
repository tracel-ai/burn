//! Cargo runner for Linux GPU CI: tolerate SIGSEGV only after libtest succeeds.
//!
//! Handling each executable separately lets Cargo run the remaining test binaries.
//! TODO: Investigate and fix GPU shutdown, then remove this workaround.
//! Related shutdown report: https://github.com/gfx-rs/wgpu/issues/8365

use std::{
    ffi::OsString,
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
};

use regex::bytes::Regex;
use tracel_xtask::prelude::{anyhow, clap};

#[derive(clap::Args)]
pub struct WgpuTestRunnerArgs {
    #[arg(required = true, num_args = 1.., trailing_var_arg = true, allow_hyphen_values = true)]
    command: Vec<OsString>,
}

pub fn run(args: &WgpuTestRunnerArgs) -> anyhow::Result<i32> {
    // libtest's color reset can include a character-set reset (ESC ( B).
    let ansi = Regex::new(r"\x1b(?:\[[0-9;]*m|\([A-Z0-9])")?;
    let start = Regex::new(r"^running (\d+) tests?$")?;
    let success = Regex::new(
        r"^test result: ok\. (\d+) passed; 0 failed; (\d+) ignored; (\d+) measured; \d+ filtered out; finished in [0-9.]+s$",
    )?;
    let mut expected = None;
    let mut completed = false;
    let mut failed = false;

    let mut child = Command::new(&args.command[0])
        .args(&args.command[1..])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut reader = BufReader::new(child.stdout.take().expect("stdout is piped"));
    let mut stdout = std::io::stdout().lock();
    let mut line = Vec::new();
    while reader.read_until(b'\n', &mut line)? != 0 {
        stdout.write_all(&line)?;
        stdout.flush()?;
        let clean = ansi.replace_all(&line, &b""[..]);
        let clean = clean.trim_ascii();
        if let Some(start) = start.captures(clean) {
            expected = Some(std::str::from_utf8(&start[1])?.parse::<u64>()?);
            completed = false;
        } else if clean.starts_with(b"test result:") {
            completed = false;
            if let Some(summary) = success.captures(clean) {
                let mut total = 0u64;
                for count in summary.iter().skip(1).flatten() {
                    let count = std::str::from_utf8(count.as_bytes())?.parse::<u64>()?;
                    total = total
                        .checked_add(count)
                        .ok_or_else(|| anyhow::anyhow!("Test count overflow"))?;
                }
                completed = expected == Some(total);
            }
            failed |= !completed;
        }
        line.clear();
    }
    let status = child.wait()?;

    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt;

        status.signal()
    };
    #[cfg(not(unix))]
    let signal: Option<i32> = None;

    if let Some(signal) = signal {
        if signal == 11 && completed && !failed {
            eprintln!(
                "::warning::Tolerated SIGSEGV after successful test completion: {}. GPU shutdown still needs investigation.",
                args.command[0].to_string_lossy()
            );
            return Ok(0);
        }
        return Ok(128 + signal);
    }
    Ok(status.code().unwrap_or(1))
}
