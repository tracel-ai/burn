use std::sync::mpsc;
use std::thread;

use ratatui::widgets::ListState;

use crate::run_support::{
    BACKENDS, ProblemKind, RunMsg, RunView, is_release, now_millis, stream_command,
};
use crate::{DTYPE_NAMES, list_runs, load_events, run_log_path, runs_dir};

pub struct App {
    pub runs: Vec<RunView>,
    pub run_list_state: ListState,

    pub backend_idx: usize,
    pub problem_idx: usize,
    pub in_dtype_idx: usize,
    pub out_dtype_idx: usize,

    pub run_rx: Option<mpsc::Receiver<RunMsg>>,
    pub pending_run: Option<String>,
    pub output_lines: Vec<String>,
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
            pending_run: None,
            output_lines: Vec::new(),
        };
        app.rescan_runs(None);
        app
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

        let mut cargo_args: Vec<String> = vec!["run".into()];
        if is_release() {
            cargo_args.push("--release".into());
        }
        cargo_args.extend(["--bin", "runner", "--features", feature].map(String::from));
        cargo_args.push("--".into());

        let stamp = now_millis();
        let shape_name = "512x512x512";
        let id = format!(
            "{stamp}-{backend}-{}-{}-{input}-{output}",
            problem.name(),
            shape_name
        );
        let run_dir = runs_dir().join(&id);

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

        self.pending_run = Some(id);
        self.output_lines.clear();
        self.output_lines
            .push(format!("$ cargo {}", cargo_args.join(" ")));

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || stream_command(cargo_args, tx));
        self.run_rx = Some(rx);
    }
}
