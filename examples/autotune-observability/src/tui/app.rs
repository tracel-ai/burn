use std::collections::VecDeque;
use std::sync::mpsc;
use std::thread;

use ratatui::widgets::ListState;

use crate::remote::{RemoteConfig, RemoteRun, run_remote};
use crate::run_support::{
    BACKENDS, ProblemKind, RunBook, RunBooks, RunMsg, RunSpec, RunView, is_release, now_millis,
    stream_command,
};
use crate::{DTYPE_NAMES, list_runs, load_events, run_log_path, runs_dir};

/// Which remote text field the TUI is currently editing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RemoteField {
    Host,
    Base,
    Password,
}

pub(crate) struct App {
    pub(crate) runs: Vec<RunView>,
    pub(crate) run_list_state: ListState,

    pub(crate) backend_idx: usize,
    pub(crate) problem_idx: usize,
    pub(crate) in_dtype_idx: usize,
    pub(crate) out_dtype_idx: usize,

    pub(crate) run_rx: Option<mpsc::Receiver<RunMsg>>,
    pub(crate) run_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub(crate) pending_run: Option<String>,
    pub(crate) output_lines: Vec<String>,

    pub(crate) shape_dims: Vec<String>,
    pub(crate) active_dim_idx: usize,
    pub(crate) input_mode: bool,
    pub(crate) events_scroll: u16,

    pub(crate) remote: RemoteConfig,
    pub(crate) remote_edit: Option<RemoteField>,
    pub(crate) force_sync: bool,
    pub(crate) disable_throughput_cache: bool,
    pub(crate) status: String,

    pub(crate) run_books: RunBooks,
    pub(crate) selected_book: usize,
    pub(crate) book_entry_state: ListState,
    /// Whether keystrokes are currently renaming the active book.
    pub(crate) editing_book_name: bool,
    /// Entries queued by "Run all", launched one after another as each run finishes.
    pub(crate) run_queue: VecDeque<QueuedLaunch>,
    /// Whether the keybinding help overlay is shown.
    pub(crate) show_help: bool,
}

/// A fully-resolved launch queued by "Run all".
pub(crate) struct QueuedLaunch {
    backend: String,
    feature: String,
    problem: ProblemKind,
    input: String,
    output: String,
    shapes: Vec<Vec<usize>>,
}

impl App {
    pub fn new() -> Self {
        let mut app = Self {
            runs: Vec::new(),
            run_list_state: ListState::default(),
            backend_idx: 0,
            problem_idx: 0,
            in_dtype_idx: 0,
            out_dtype_idx: 0,
            run_rx: None,
            run_cancel: None,
            pending_run: None,
            output_lines: Vec::new(),
            shape_dims: ProblemKind::Matmul.default_shape().into_iter().map(|s| s.to_string()).collect(),
            active_dim_idx: 0,
            input_mode: false,
            events_scroll: 0,
            remote: RemoteConfig::load(),
            remote_edit: None,
            force_sync: false,
            disable_throughput_cache: false,
            status: String::from("Ready."),
            run_books: RunBooks::load(),
            selected_book: 0,
            book_entry_state: ListState::default(),
            editing_book_name: false,
            run_queue: VecDeque::new(),
            show_help: false,
        };
        app.rescan_runs(None);
        app.sync_book_selection();
        app
    }

    /// Mutable access to the remote text field being edited.
    pub(crate) fn remote_field_mut(&mut self, field: RemoteField) -> &mut String {
        match field {
            RemoteField::Host => &mut self.remote.host,
            RemoteField::Base => &mut self.remote.base_dir,
            RemoteField::Password => &mut self.remote.password,
        }
    }

    pub fn rescan_runs(&mut self, select: Option<&str>) {
        let selected_name = self
            .run_list_state
            .selected()
            .and_then(|i| self.runs.get(i))
            .map(|r| r.name.clone());
        let to_select = select.map(|s| s.to_string()).or(selected_name);

        self.runs = list_runs()
            .into_iter()
            .map(|dir| {
                let name = dir.file_name().unwrap().to_string_lossy().into_owned();
                let events = load_events(&run_log_path(&dir));
                let custom_name = std::fs::read_to_string(dir.join("name.txt"))
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                RunView {
                    name,
                    dir,
                    events,
                    selected: false,
                    custom_name,
                }
            })
            .collect();

        if !self.runs.is_empty() {
            let mut idx = 0;
            if let Some(target) = to_select {
                if let Some(pos) = self.runs.iter().position(|r| r.name == target) {
                    idx = pos;
                }
            }
            self.run_list_state.select(Some(idx));
        } else {
            self.run_list_state.select(None);
        }
    }

    pub fn run_selected_problem(&mut self) {
        let (_, backend, feature) = BACKENDS[self.backend_idx];
        let input = DTYPE_NAMES[self.in_dtype_idx].to_string();
        let output = DTYPE_NAMES[self.out_dtype_idx].to_string();
        let problem = ProblemKind::ALL[self.problem_idx];
        let shape: Vec<usize> = self.shape_dims.iter().map(|d| d.parse().unwrap_or(0)).collect();
        self.launch(backend, feature, problem, input, output, vec![shape]);
    }

    /// Spawn one runner process for the given config (which may cover several shapes). No-op if a
    /// run is already in flight.
    fn launch(
        &mut self,
        backend: &str,
        feature: &str,
        problem: ProblemKind,
        input: String,
        output: String,
        shapes: Vec<Vec<usize>>,
    ) {
        if self.run_rx.is_some() {
            return;
        }

        let shape_names: Vec<String> = shapes
            .iter()
            .map(|shape| shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"))
            .collect();
        let stamp = now_millis();
        let id = format!(
            "{stamp}-{backend}-{}-{}-{input}-{output}",
            problem.name(),
            shape_names.join("_")
        );
        let run_dir = runs_dir().join(&id);

        self.pending_run = Some(id.clone());
        self.output_lines.clear();

        let (tx, rx) = mpsc::channel();
        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.run_cancel = Some(std::sync::Arc::clone(&cancel_flag));

        if self.remote.enabled {
            self.output_lines
                .push(format!("$ remote run on {}", self.remote.host));
            self.status = format!("Running on {}…", self.remote.host);
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
            cargo_args.extend([
                "--backend".into(),
                backend.to_string(),
                "--problem".into(),
                problem.name().into(),
                "--input".into(),
                input,
                "--output".into(),
                output,
                "--run-dir".into(),
                run_dir.to_string_lossy().into_owned(),
            ]);
            for shape_name in &shape_names {
                cargo_args.push("--shape".into());
                cargo_args.push(shape_name.clone());
            }
            if self.disable_throughput_cache {
                cargo_args.push("--no-throughput-cache".into());
            }
            self.output_lines
                .push(format!("$ cargo {}", cargo_args.join(" ")));
            self.status = String::from("Running locally…");
            thread::spawn(move || stream_command(cargo_args, tx, cancel_flag));
        }

        self.run_rx = Some(rx);
    }

    pub fn cancel_run(&mut self) {
        self.run_queue.clear();
        if let Some(flag) = &self.run_cancel {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Launch the next queued entry, if any. Returns whether one was started.
    pub(crate) fn start_next_queued(&mut self) -> bool {
        while let Some(queued) = self.run_queue.pop_front() {
            let QueuedLaunch {
                backend,
                feature,
                problem,
                input,
                output,
                shapes,
            } = queued;
            self.launch(&backend, &feature, problem, input, output, shapes);
            if self.run_rx.is_some() {
                let remaining = self.run_queue.len();
                if remaining > 0 {
                    self.status = format!("{} ({remaining} queued)", self.status);
                }
                return true;
            }
        }
        false
    }

    // --- Run books -------------------------------------------------------------------------

    pub(crate) fn active_book_index(&self) -> usize {
        self.selected_book.min(self.run_books.books.len() - 1)
    }

    pub(crate) fn active_book(&self) -> &RunBook {
        &self.run_books.books[self.active_book_index()]
    }

    fn active_book_mut(&mut self) -> &mut RunBook {
        let index = self.active_book_index();
        &mut self.run_books.books[index]
    }

    /// Keep `selected_book` and the entry selection in range after the books change.
    fn sync_book_selection(&mut self) {
        self.selected_book = self.active_book_index();
        let len = self.active_book().specs.len();
        if len == 0 {
            self.book_entry_state.select(None);
        } else {
            let idx = self.book_entry_state.selected().unwrap_or(0).min(len - 1);
            self.book_entry_state.select(Some(idx));
        }
    }

    pub(crate) fn select_next_book(&mut self) {
        self.selected_book = (self.active_book_index() + 1) % self.run_books.books.len();
        self.sync_book_selection();
    }

    pub(crate) fn select_prev_book(&mut self) {
        let len = self.run_books.books.len();
        self.selected_book = (self.active_book_index() + len - 1) % len;
        self.sync_book_selection();
    }

    pub(crate) fn new_book(&mut self) {
        let mut n = self.run_books.books.len() + 1;
        let name = loop {
            let candidate = format!("Book {n}");
            if !self.run_books.books.iter().any(|book| book.name == candidate) {
                break candidate;
            }
            n += 1;
        };
        let backend = BACKENDS[self.backend_idx].1.to_string();
        self.run_books.books.push(RunBook::new(name, backend));
        self.selected_book = self.run_books.books.len() - 1;
        self.run_books.save();
        self.sync_book_selection();
        self.status = String::from("Created a new book.");
    }

    pub(crate) fn delete_book(&mut self) {
        let index = self.active_book_index();
        self.run_books.books.remove(index);
        if self.run_books.books.is_empty() {
            self.run_books
                .books
                .push(RunBook::new("Default", BACKENDS[0].1.to_string()));
        }
        self.selected_book = self.selected_book.min(self.run_books.books.len() - 1);
        self.run_books.save();
        self.sync_book_selection();
        self.status = String::from("Deleted book.");
    }

    /// Cycle the active book's default backend to the next one.
    pub(crate) fn cycle_book_backend(&mut self) {
        let current = self.active_book().backend.clone();
        let pos = BACKENDS.iter().position(|(_, b, _)| *b == current).unwrap_or(0);
        let next = BACKENDS[(pos + 1) % BACKENDS.len()].1.to_string();
        self.active_book_mut().backend = next;
        self.run_books.save();
    }

    /// Cycle the selected entry's backend override: book default → each backend → back to default.
    pub(crate) fn cycle_selected_entry_backend(&mut self) {
        let Some(idx) = self.book_entry_state.selected() else {
            return;
        };
        let book = self.active_book_mut();
        let Some(spec) = book.specs.get_mut(idx) else {
            return;
        };
        spec.backend = match &spec.backend {
            None => Some(BACKENDS[0].1.to_string()),
            Some(current) => {
                let pos = BACKENDS.iter().position(|(_, b, _)| b == current).unwrap_or(0);
                if pos + 1 >= BACKENDS.len() {
                    None
                } else {
                    Some(BACKENDS[pos + 1].1.to_string())
                }
            }
        };
        self.run_books.save();
    }

    pub(crate) fn book_entry_next(&mut self) {
        let len = self.active_book().specs.len();
        if len == 0 {
            return;
        }
        let idx = self.book_entry_state.selected().map_or(0, |i| (i + 1) % len);
        self.book_entry_state.select(Some(idx));
    }

    pub(crate) fn book_entry_prev(&mut self) {
        let len = self.active_book().specs.len();
        if len == 0 {
            return;
        }
        let idx = self
            .book_entry_state
            .selected()
            .map_or(0, |i| (i + len - 1) % len);
        self.book_entry_state.select(Some(idx));
    }

    /// Add the current controls (problem/dtypes/shapes) as an entry in the active book. The current
    /// backend becomes a per-entry override only when it differs from the book default.
    pub(crate) fn book_add_current(&mut self) {
        let current_backend = BACKENDS[self.backend_idx].1.to_string();
        let backend = (current_backend != self.active_book().backend).then_some(current_backend);
        let shape: Vec<usize> = self.shape_dims.iter().map(|d| d.parse().unwrap_or(0)).collect();
        let spec = RunSpec {
            name: String::new(),
            backend,
            problem: ProblemKind::ALL[self.problem_idx],
            input: DTYPE_NAMES[self.in_dtype_idx].to_string(),
            output: DTYPE_NAMES[self.out_dtype_idx].to_string(),
            shapes: vec![shape],
        };
        self.active_book_mut().specs.push(spec);
        self.run_books.save();
        self.sync_book_selection();
        self.status = String::from("Added current config to the book.");
    }

    pub(crate) fn book_delete_selected_entry(&mut self) {
        let Some(idx) = self.book_entry_state.selected() else {
            return;
        };
        let book = self.active_book_mut();
        if idx < book.specs.len() {
            book.specs.remove(idx);
            self.run_books.save();
            self.sync_book_selection();
            self.status = String::from("Deleted entry.");
        }
    }

    /// Queue every entry in the active book and launch them one after another.
    pub(crate) fn book_run_all(&mut self) {
        if self.run_rx.is_some() {
            return;
        }
        let book = self.active_book();
        let mut queue = VecDeque::new();
        let mut skipped = 0;
        for spec in &book.specs {
            let backend = book.effective_backend(spec);
            match BACKENDS.iter().find(|(_, b, _)| *b == backend) {
                Some((_, _, feature)) => queue.push_back(QueuedLaunch {
                    backend,
                    feature: feature.to_string(),
                    problem: spec.problem,
                    input: spec.input.clone(),
                    output: spec.output.clone(),
                    shapes: spec.shapes.clone(),
                }),
                None => skipped += 1,
            }
        }
        if queue.is_empty() {
            self.status = String::from("Nothing to run in this book.");
            return;
        }
        let total = queue.len();
        self.run_queue = queue;
        self.status = if skipped > 0 {
            format!("Running all {total} entries ({skipped} skipped: unknown backend)…")
        } else {
            format!("Running all {total} entries…")
        };
        self.start_next_queued();
    }

    pub(crate) fn book_run_selected_entry(&mut self) {
        if self.run_rx.is_some() {
            return;
        }
        let Some(idx) = self.book_entry_state.selected() else {
            return;
        };
        let book = self.active_book();
        let Some(spec) = book.specs.get(idx) else {
            return;
        };
        let backend = book.effective_backend(spec);
        let Some(feature) = BACKENDS
            .iter()
            .find(|(_, b, _)| *b == backend)
            .map(|(_, _, feature)| feature.to_string())
        else {
            self.status = format!("Unknown backend '{backend}' in entry.");
            return;
        };
        let problem = spec.problem;
        let input = spec.input.clone();
        let output = spec.output.clone();
        let shapes = spec.shapes.clone();
        self.launch(&backend, &feature, problem, input, output, shapes);
    }
}
