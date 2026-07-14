use std::path::Path;
use std::sync::mpsc::{self};
use std::thread;

use eframe::egui;
use egui::FontId;
use egui::text::TextFormat;

use super::{AutotuneObservabilityApp, LaunchRequest};
use crate::ansi::{self};
use crate::remote::{RemoteRun, run_remote, test_connection};
use crate::run_support::{
    BACKENDS, ProblemKind, RunBook, RunMsg, RunSpec, RunView, ansi_color, is_release, now_millis,
    stream_command,
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
        self.rerun_queue.clear();
        let request = LaunchRequest {
            backend: BACKENDS[self.selected].1.to_string(),
            problem: self.problem,
            input: DTYPE_NAMES[self.input_dtype].to_string(),
            output: DTYPE_NAMES[self.output_dtype].to_string(),
            shapes: self.shapes.clone(),
            source_label: None,
        };
        if let Err(err) = self.start_run_request(request) {
            self.status = err;
        }
    }

    /// The index of the currently shown book, clamped into range (books are never empty).
    fn active_book_index(&self) -> usize {
        self.selected_book.min(self.run_books.books.len() - 1)
    }

    pub(super) fn active_book(&self) -> &RunBook {
        &self.run_books.books[self.active_book_index()]
    }

    fn active_book_mut(&mut self) -> &mut RunBook {
        let index = self.active_book_index();
        &mut self.run_books.books[index]
    }

    /// Build a launch request from a saved spec, resolving its backend against the book default and
    /// labelling it by the spec's name (or a fallback).
    fn spec_to_request(spec: &RunSpec, backend: String) -> LaunchRequest {
        let label = if spec.name.trim().is_empty() {
            format!("{} {}", spec.problem.label(), backend)
        } else {
            spec.name.clone()
        };
        LaunchRequest {
            backend,
            problem: spec.problem,
            input: spec.input.clone(),
            output: spec.output.clone(),
            shapes: spec.shapes.clone(),
            source_label: Some(label),
        }
    }

    /// Snapshot the current controls (problem/dtypes/shapes) as a new entry in the book. The
    /// current backend becomes a per-entry override only when it differs from the book default.
    pub(super) fn add_current_to_run_book(&mut self) {
        let current_backend = BACKENDS[self.selected].1.to_string();
        let backend = (current_backend != self.active_book().backend).then_some(current_backend);
        let spec = RunSpec {
            name: String::new(),
            backend,
            problem: self.problem,
            input: DTYPE_NAMES[self.input_dtype].to_string(),
            output: DTYPE_NAMES[self.output_dtype].to_string(),
            shapes: self.shapes.clone(),
        };
        let book = self.active_book_mut();
        book.specs.push(spec);
        let book_name = book.name.clone();
        self.run_books.save();
        self.status = format!("Added current config to '{book_name}'.");
    }

    /// Queue every entry in the active book and launch them one after another.
    pub(super) fn run_book_run_all(&mut self) {
        let book = self.active_book();
        let requests: Vec<LaunchRequest> = book
            .specs
            .iter()
            .map(|spec| Self::spec_to_request(spec, book.effective_backend(spec)))
            .collect();
        if requests.is_empty() {
            self.status = String::from("This book is empty.");
            return;
        }
        self.start_rerun_requests(requests);
    }

    /// Launch a single entry from the active book.
    pub(super) fn run_book_run_one(&mut self, index: usize) {
        let book = self.active_book();
        let Some(spec) = book.specs.get(index) else {
            return;
        };
        let request = Self::spec_to_request(spec, book.effective_backend(spec));
        self.rerun_queue.clear();
        if let Err(err) = self.start_run_request(request) {
            self.status = err;
        }
    }

    /// Copy a book entry back into the top controls so it can be tweaked and re-saved.
    pub(super) fn load_spec_into_controls(&mut self, index: usize) {
        let book = self.active_book();
        let Some(spec) = book.specs.get(index).cloned() else {
            return;
        };
        let backend = book.effective_backend(&spec);
        if let Some(pos) = BACKENDS.iter().position(|(_, name, _)| *name == backend) {
            self.selected = pos;
        }
        self.problem = spec.problem;
        if let Some(pos) = DTYPE_NAMES.iter().position(|name| *name == spec.input) {
            self.input_dtype = pos;
        }
        if let Some(pos) = DTYPE_NAMES.iter().position(|name| *name == spec.output) {
            self.output_dtype = pos;
        }
        if !spec.shapes.is_empty() {
            self.shapes = spec.shapes;
        }
        self.status = format!("Loaded '{}' into the controls.", spec.name);
    }

    pub(super) fn duplicate_run_book_spec(&mut self, index: usize) {
        let book = self.active_book_mut();
        if let Some(spec) = book.specs.get(index).cloned() {
            book.specs.insert(index + 1, spec);
            self.run_books.save();
        }
    }

    pub(super) fn delete_run_book_spec(&mut self, index: usize) {
        let book = self.active_book_mut();
        if index < book.specs.len() {
            book.specs.remove(index);
            self.run_books.save();
        }
        self.run_book_editing = None;
    }

    /// Create a fresh empty book with a unique name and switch to it.
    pub(super) fn new_run_book(&mut self) {
        let mut n = self.run_books.books.len() + 1;
        let name = loop {
            let candidate = format!("Book {n}");
            if !self.run_books.books.iter().any(|book| book.name == candidate) {
                break candidate;
            }
            n += 1;
        };
        let backend = BACKENDS[self.selected].1.to_string();
        self.run_books.books.push(RunBook::new(name, backend));
        self.selected_book = self.run_books.books.len() - 1;
        self.run_book_editing = None;
        self.run_books.save();
    }

    /// Delete the active book, keeping at least one book around.
    pub(super) fn delete_active_book(&mut self) {
        let index = self.active_book_index();
        self.run_books.books.remove(index);
        if self.run_books.books.is_empty() {
            self.run_books.books.push(RunBook::new("Default", BACKENDS[0].1.to_string()));
        }
        self.selected_book = self.selected_book.min(self.run_books.books.len() - 1);
        self.run_book_editing = None;
        self.run_books.save();
    }

    pub(super) fn select_book(&mut self, index: usize) {
        if index < self.run_books.books.len() && index != self.selected_book {
            self.selected_book = index;
            self.run_book_editing = None;
        }
    }

    pub(super) fn save_run_books(&self) {
        self.run_books.save();
    }

    pub(super) fn rerun_selected_runs(&mut self) {
        self.rerun_popup = None;
        if let Some(requests) = self.collect_selected_rerun_requests() {
            self.start_rerun_requests(requests);
        }
    }

    pub(super) fn open_rerun_selected_popup(&mut self) {
        self.rerun_popup = self.collect_selected_rerun_requests();
    }
    fn collect_selected_rerun_requests(&mut self) -> Option<Vec<LaunchRequest>> {
        self.rerun_queue.clear();

        let selected: Vec<(String, String, std::path::PathBuf)> = self
            .runs
            .iter()
            .filter(|run| run.selected)
            .map(|run| {
                (
                    run.name.clone(),
                    run.custom_name.clone().unwrap_or_else(|| run.name.clone()),
                    run.dir.clone(),
                )
            })
            .collect();

        let mut requests = Vec::new();
        let mut skipped = Vec::new();
        for (name, label, dir) in selected {
            match Self::load_launch_request(&name, &label, &dir) {
                Ok(request) => requests.push(request),
                Err(err) => skipped.push(format!("{label}: {err}")),
            }
        }

        if requests.is_empty() {
            self.status = if skipped.is_empty() {
                String::from("No selected runs to rerun.")
            } else {
                format!("Couldn't rerun selected runs: {}", skipped.join("; "))
            };
            return None;
        }

        if !skipped.is_empty() {
            for message in skipped {
                self.push_line(&format!("[rerun skipped] {message}"));
            }
        }

        Some(requests)
    }

    pub(super) fn start_rerun_requests(&mut self, mut requests: Vec<LaunchRequest>) {
        let total = requests.len();
        let first = requests.remove(0);
        self.rerun_queue = requests.into();
        self.push_line(&format!("[rerun] queued {total} selected run(s)"));
        if let Err(err) = self.start_run_request(first) {
            self.status = err;
            self.start_next_queued_run();
        }
    }
    fn start_run_request(&mut self, request: LaunchRequest) -> Result<(), String> {
        let Some((_, backend, feature)) = BACKENDS
            .iter()
            .copied()
            .find(|(_, backend, _)| *backend == request.backend)
        else {
            return Err(format!(
                "Unknown backend '{}' in saved run",
                request.backend
            ));
        };

        if !DTYPE_NAMES.contains(&request.input.as_str()) {
            return Err(format!(
                "Unknown input dtype '{}' in saved run",
                request.input
            ));
        }
        if !DTYPE_NAMES.contains(&request.output.as_str()) {
            return Err(format!(
                "Unknown output dtype '{}' in saved run",
                request.output
            ));
        }
        if request.shapes.is_empty() {
            return Err(String::from("Saved run has no shapes to replay"));
        }

        let problem = request.problem;
        let input = request.input;
        let output = request.output;
        let shapes = request.shapes;
        let queued_remaining = self.rerun_queue.len();
        if let Some(source) = &request.source_label {
            self.push_line(&format!("[rerun] {source}"));
        }

        let shape_names: Vec<String> = shapes
            .iter()
            .map(|shape| {
                shape
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            })
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
        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.run_cancel = Some(std::sync::Arc::clone(&cancel_flag));

        let queue_suffix = match queued_remaining {
            0 => String::new(),
            1 => String::from(" (1 rerun queued)"),
            n => format!(" ({n} reruns queued)"),
        };

        if self.remote.enabled {
            self.push_line(&format!("$ remote run on {}", self.remote.host));
            self.status = format!(
                "Running {} on {}{}…",
                problem.label().to_lowercase(),
                self.remote.host,
                queue_suffix
            );
            let cfg = self.remote.clone();
            let run = RemoteRun {
                feature: feature.to_string(),
                backend: backend.to_string(),
                problem,
                input,
                output,
                shapes,
                id,
                local_run_dir: run_dir,
                force_sync: self.force_sync,
                disable_throughput_cache: self.disable_throughput_cache,
            };
            thread::spawn(move || run_remote(cfg, run, cancel_flag, tx));
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
                    &input,
                    "--output",
                    &output,
                ]
                .map(String::from),
            );
            for shape in &shapes {
                cargo_args.extend([
                    "--shape".to_string(),
                    shape
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join("x"),
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
                "Running {}{}… (first run of a backend compiles it)",
                problem.label().to_lowercase(),
                queue_suffix
            );
            thread::spawn(move || stream_command(cargo_args, tx, cancel_flag));
        }

        self.run_rx = Some(rx);
        Ok(())
    }

    fn start_next_queued_run(&mut self) -> bool {
        while let Some(request) = self.rerun_queue.pop_front() {
            match self.start_run_request(request) {
                Ok(()) => return true,
                Err(err) => self.push_line(&format!("[rerun skipped] {err}")),
            }
        }
        false
    }

    fn load_launch_request(name: &str, label: &str, dir: &Path) -> Result<LaunchRequest, String> {
        Self::load_launch_request_from_meta(dir)
            .or_else(|meta_err| {
                Self::load_launch_request_from_name(name)
                    .map_err(|name_err| format!("{meta_err}; fallback parse failed: {name_err}"))
            })
            .map(|mut request| {
                request.source_label = Some(label.to_string());
                request
            })
    }

    fn load_launch_request_from_meta(dir: &Path) -> Result<LaunchRequest, String> {
        let text = std::fs::read_to_string(dir.join("meta.txt"))
            .map_err(|err| format!("missing meta.txt: {err}"))?;
        let mut backend = None;
        let mut problem = None;
        let mut shapes = None;
        let mut input = None;
        let mut output = None;

        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            if let Some(value) = line.strip_prefix("backend=") {
                backend = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("problem=") {
                problem = Some(ProblemKind::from_str(value)?);
            } else if let Some(value) = line.strip_prefix("shapes=") {
                shapes = Some(
                    value
                        .split(',')
                        .map(Self::parse_shape)
                        .collect::<Result<Vec<_>, _>>()?,
                );
            } else if let Some(value) = line.strip_prefix("input=") {
                let (in_dtype, out_dtype) = value
                    .split_once(" output=")
                    .ok_or_else(|| format!("invalid input/output line '{line}'"))?;
                input = Some(in_dtype.to_string());
                output = Some(out_dtype.to_string());
            }
        }

        Ok(LaunchRequest {
            backend: backend.ok_or_else(|| String::from("missing backend"))?,
            problem: problem.ok_or_else(|| String::from("missing problem"))?,
            input: input.ok_or_else(|| String::from("missing input dtype"))?,
            output: output.ok_or_else(|| String::from("missing output dtype"))?,
            shapes: shapes.ok_or_else(|| String::from("missing shapes"))?,
            source_label: None,
        })
    }

    fn load_launch_request_from_name(name: &str) -> Result<LaunchRequest, String> {
        let parts: Vec<&str> = name.split('-').collect();
        if parts.len() < 6 {
            return Err(format!("run name '{name}' doesn't match archived format"));
        }

        let shapes_idx = parts.len() - 3;
        let problem = parts[2..shapes_idx].join("-");
        Ok(LaunchRequest {
            backend: parts[1].to_string(),
            problem: ProblemKind::from_str(&problem)?,
            input: parts[parts.len() - 2].to_string(),
            output: parts[parts.len() - 1].to_string(),
            shapes: parts[shapes_idx]
                .split('_')
                .map(Self::parse_shape)
                .collect::<Result<Vec<_>, _>>()?,
            source_label: None,
        })
    }

    fn parse_shape(value: &str) -> Result<Vec<usize>, String> {
        value
            .split('x')
            .map(|dim| {
                dim.parse::<usize>()
                    .map_err(|_| format!("invalid shape component '{dim}' in '{value}'"))
            })
            .collect()
    }

    pub(super) fn cancel_run(&mut self) {
        self.rerun_queue.clear();
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
                if self.start_next_queued_run() {
                    ctx.request_repaint();
                }
            }
            Some(false) => {
                self.run_rx = None;
                self.pending_run = None;
                self.status = String::from("Run failed — see output panel.");
                if self.start_next_queued_run() {
                    ctx.request_repaint();
                }
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
