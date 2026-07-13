use std::sync::mpsc::{self};
use std::thread;

use eframe::egui;
use egui::FontId;
use egui::text::TextFormat;

use super::AutotuneObservabilityApp;
use crate::ansi::{self};
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
                RunView {
                    name,
                    dir,
                    events,
                    selected,
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
    pub(super) fn run_matmul(&mut self) {
        let (_, backend, feature) = BACKENDS[self.selected];
        let input = DTYPE_NAMES[self.input_dtype];
        let output = DTYPE_NAMES[self.output_dtype];

        let mut cargo_args: Vec<String> = vec!["run".into()];
        if is_release() {
            cargo_args.push("--release".into());
        }
        cargo_args.extend(["--bin", "runner", "--features", feature].map(String::from));
        cargo_args.push("--".into());

        let shape_names: Vec<String> = self
            .matmuls
            .iter()
            .map(|shape| format!("{}x{}x{}", shape.m, shape.k, shape.n))
            .collect();
        let stamp = now_millis();
        let id = format!(
            "{stamp}-{backend}-{}-{input}-{output}",
            shape_names.join("_")
        );
        let run_dir = runs_dir().join(&id);

        cargo_args
            .extend(["--backend", backend, "--input", input, "--output", output].map(String::from));
        for shape in &self.matmuls {
            cargo_args.extend([
                "--matmul".to_string(),
                format!("{}x{}x{}", shape.m, shape.k, shape.n),
            ]);
        }
        cargo_args.extend([
            "--run-dir".to_string(),
            run_dir.to_string_lossy().into_owned(),
        ]);

        self.pending_run = Some(id);

        self.ansi_style = Default::default();
        self.push_line(&format!("$ cargo {}", cargo_args.join(" ")));
        self.status = String::from("Running… (first run of a backend compiles it)");

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || stream_command(cargo_args, tx));
        self.run_rx = Some(rx);
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
        let mut finished = None;
        loop {
            match rx.try_recv() {
                Ok(RunMsg::Line(line)) => lines.push(line),
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

        match finished {
            Some(true) => {
                self.run_rx = None;
                let id = self.pending_run.take();
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
}
