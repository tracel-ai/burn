use std::sync::mpsc::{self};
use std::thread;

use eframe::egui;
use egui::FontId;
use egui::text::TextFormat;

use super::AutotuneObservabilityApp;
use crate::ansi::{self};
use crate::remote::{RemoteRun, run_remote, test_connection};
use crate::run_support::{
    BACKENDS, RunMsg, RunView, ansi_color, is_release, now_millis, stream_command,
};
use crate::{DTYPE_NAMES, list_runs, load_events, run_log_path, runs_dir};

impl AutotuneObservabilityApp {
    /// Rebuild the run list from disk, preserving which runs are shown (by name) and optionally
    /// forcing `select` to be shown.
    pub(super) fn rescan_runs(&mut self, select: Option<&str>) {
        let shown: Vec<String> = self
            .runs
            .iter()
            .filter(|r| r.selected)
            .map(|r| r.name.clone())
            .collect();

        self.runs = list_runs()
            .into_iter()
            .map(|dir| {
                let name = dir
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let events = load_events(&run_log_path(&dir));
                let selected = Some(name.as_str()) == select || shown.contains(&name);
                let custom_name = std::fs::read_to_string(dir.join("name.txt"))
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                RunView {
                    name,
                    dir,
                    events,
                    selected,
                    custom_name,
                }
            })
            .collect();

        // Default to showing the newest run when nothing is selected.
        if !self.runs.iter().any(|r| r.selected) {
            if let Some(first) = self.runs.first_mut() {
                first.selected = true;
            }
        }
    }

    /// Launch the runner through `cargo run` on a worker thread. Cargo builds it on demand (in
    /// the UI's profile, with the selected backend's feature), and stdout/stderr — build
    /// progress and the run itself — stream back line by line. A fresh process per run means an
    /// empty in-memory autotune cache, so the same config re-tunes every time.
    pub(super) fn run_selected_problem(&mut self) {
        let (_, backend, feature) = BACKENDS[self.selected];
        let input = DTYPE_NAMES[self.input_dtype];
        let output = DTYPE_NAMES[self.output_dtype];
        let problem = self.problem;

        let shape_names: Vec<String> = self
            .shapes
            .iter()
            .map(|shape| format!("{}x{}x{}", shape.m, shape.k, shape.n))
            .collect();
        let stamp = now_millis();
        let id = format!(
            "{stamp}-{backend}-{}-{}-{input}-{output}",
            problem.name(),
            shape_names.join("_")
        );
        let run_dir = runs_dir().join(&id);

        self.pending_run = Some(id.clone());
        self.ansi_style = Default::default();

        let (tx, rx) = mpsc::channel();
        self.run_cancel = None;

        if self.remote.enabled {
            self.push_line(&format!("$ remote run on {}", self.remote.host));
            self.status = format!(
                "Running {} on {} (syncing + building on first run)…",
                problem.label().to_lowercase(),
                self.remote.host
            );
            let cfg = self.remote.clone();
            let run = RemoteRun {
                feature: feature.to_string(),
                backend: backend.to_string(),
                problem,
                input: input.to_string(),
                output: output.to_string(),
                shapes: self.shapes.iter().map(|s| (s.m, s.k, s.n)).collect(),
                id,
                local_run_dir: run_dir,
                force_sync: self.force_sync,
                disable_throughput_cache: self.disable_throughput_cache,
            };
            thread::spawn(move || run_remote(cfg, run, tx));
        } else {
            let mut cargo_args: Vec<String> = vec!["run".into()];
            if is_release() {
                cargo_args.push("--release".into());
            }
            cargo_args.extend(["--bin", "runner", "--features", feature].map(String::from));
            cargo_args.push("--".into());
            cargo_args.extend(
                [
                    "--backend",
                    backend,
                    "--problem",
                    problem.name(),
                    "--input",
                    input,
                    "--output",
                    output,
                ]
                .map(String::from),
            );
            for shape in &self.shapes {
                cargo_args.extend([
                    "--shape".to_string(),
                    format!("{}x{}x{}", shape.m, shape.k, shape.n),
                ]);
            }
            cargo_args.extend([
                "--run-dir".to_string(),
                run_dir.to_string_lossy().into_owned(),
            ]);
            if self.disable_throughput_cache {
                cargo_args.push("--no-throughput-cache".to_string());
            }

            self.push_line(&format!("$ cargo {}", cargo_args.join(" ")));
            self.status = format!(
                "Running {}… (first run of a backend compiles it)",
                problem.label().to_lowercase()
            );
            let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            self.run_cancel = Some(std::sync::Arc::clone(&cancel_flag));
            thread::spawn(move || stream_command(cargo_args, tx, cancel_flag));
        }

        self.run_rx = Some(rx);
    }

    pub(super) fn cancel_run(&mut self) {
        if let Some(flag) = &self.run_cancel {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Connect to the remote on a worker thread and report the outcome in the status line.
    pub(super) fn test_remote_connection(&mut self) {
        let cfg = self.remote.clone();
        self.status = format!("Testing connection to {}…", self.remote.host);
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let message = match test_connection(&cfg) {
                Ok(message) => message,
                Err(err) => format!("Connection failed: {err}"),
            };
            let _ = tx.send(message);
        });
        self.conn_test = Some(rx);
    }

    /// Append one line to the console output, colouring its ANSI spans.
    fn push_line(&mut self, line: &str) {
        for (style, text) in ansi::parse_ansi(line, &mut self.ansi_style) {
            let color = ansi_color(style, self.text_color);
            self.output.append(
                &text,
                0.0,
                TextFormat {
                    font_id: FontId::monospace(12.0),
                    color,
                    ..Default::default()
                },
            );
        }
        self.output.append("\n", 0.0, TextFormat::default());
    }

    pub(super) fn poll_run(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.run_rx else { return };
        let mut lines = Vec::new();
        let mut progress = None;
        let mut finished = None;
        loop {
            match rx.try_recv() {
                Ok(RunMsg::Line(line)) => lines.push(line),
                Ok(RunMsg::Progress(status)) => progress = Some(status),
                Ok(RunMsg::Done { ok }) => finished = Some(ok),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    finished = finished.or(Some(false));
                    break;
                }
            }
        }
        for line in lines {
            self.push_line(&line);
        }
        if let Some(status) = progress {
            self.status = status;
        }

        match finished {
            Some(true) => {
                self.run_rx = None;
                let id = self.pending_run.take();
                self.rescan_backends();
                self.rescan_runs(id.as_deref());
                self.status = match id {
                    Some(id) => format!("Run {id} archived."),
                    None => String::from("Run finished."),
                };
            }
            Some(false) => {
                self.run_rx = None;
                self.pending_run = None;
                self.status = String::from("Run failed — see output panel.");
            }
            None => ctx.request_repaint(),
        }
    }

    pub(super) fn delete_run(&mut self, dir: &std::path::Path) {
        let _ = std::fs::remove_dir_all(dir);
        self.rescan_runs(None);
        self.status = String::from("Deleted run.");
    }

    pub(super) fn rename_run(&mut self, i: usize, new_name: String) {
        if let Some(run) = self.runs.get_mut(i) {
            let name_file = run.dir.join("name.txt");
            if new_name.is_empty() {
                let _ = std::fs::remove_file(name_file);
                run.custom_name = None;
            } else {
                let _ = std::fs::write(name_file, &new_name);
                run.custom_name = Some(new_name);
            }
        }
    }

    pub(super) fn rescan_backends(&mut self) {
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let Some(dir) = exe.parent() else { return };
        let runner_exe = dir.join("runner");
        if !runner_exe.exists() {
            self.built_backends.clear();
            return;
        }
        if let Ok(output) = std::process::Command::new(runner_exe)
            .arg("--list-backends")
            .output()
        {
            let text = String::from_utf8_lossy(&output.stdout);
            self.built_backends = text.lines().map(|s| s.trim().to_string()).collect();
        }
    }
}
