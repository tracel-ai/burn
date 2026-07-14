use std::sync::mpsc;
use std::thread;

use ratatui::widgets::ListState;

use crate::remote::{RemoteConfig, RemoteRun, run_remote};
use crate::run_support::{
    BACKENDS, ProblemKind, RunMsg, RunView, is_release, now_millis, stream_command,
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
        };
        app.rescan_runs(None);
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
        if self.run_rx.is_some() {
            return;
        }

        let (_, backend, feature) = BACKENDS[self.backend_idx];
        let input = DTYPE_NAMES[self.in_dtype_idx];
        let output = DTYPE_NAMES[self.out_dtype_idx];
        let problem = ProblemKind::ALL[self.problem_idx];

        let stamp = now_millis();
        let shape_name = self.shape_dims.join("x");
        let id = format!(
            "{stamp}-{backend}-{}-{}-{input}-{output}",
            problem.name(),
            shape_name
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
            let shape: Vec<usize> =
                self.shape_dims.iter().map(|d| d.parse().unwrap_or(0)).collect();
            let cfg = self.remote.clone();
            let run = RemoteRun {
                feature: feature.to_string(),
                backend: backend.to_string(),
                problem,
                input: input.to_string(),
                output: output.to_string(),
                shapes: vec![shape],
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
                backend.into(),
                "--problem".into(),
                problem.name().into(),
                "--input".into(),
                input.into(),
                "--output".into(),
                output.into(),
                "--shape".into(),
                shape_name.into(),
                "--run-dir".into(),
                run_dir.to_string_lossy().into_owned(),
            ]);
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
        if let Some(flag) = &self.run_cancel {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
