use std::sync::mpsc::Receiver;

use eframe::egui;
use egui::Color32;
use egui::text::LayoutJob;

use crate::ansi::AnsiStyle;
use crate::remote::RemoteConfig;
use crate::run_support::{ProblemKind, RunMsg, RunView};

mod actions;
mod panels;

pub struct AutotuneObservabilityApp {
    runs: Vec<RunView>,
    selected: usize,
    input_dtype: usize,
    output_dtype: usize,
    problem: ProblemKind,
    shapes: Vec<Vec<usize>>,
    only_short_circuits: bool,
    status: String,
    output: LayoutJob,
    ansi_style: AnsiStyle,
    text_color: Color32,
    run_rx: Option<Receiver<RunMsg>>,
    run_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Directory name of the in-flight run, selected once it finishes successfully.
    pending_run: Option<String>,
    rename_buffer: Option<(usize, String)>,
    built_backends: Vec<String>,
    comparison_mode: bool,
    remote: RemoteConfig,
    /// Receives the one-line result of an in-flight "Test connection".
    conn_test: Option<Receiver<String>>,
    /// Force the next remote run to bypass the sync cache and re-check every file.
    force_sync: bool,
    /// Re-benchmark the peak-throughput bound each run instead of reusing cubecl's global cache.
    disable_throughput_cache: bool,
}

impl Default for AutotuneObservabilityApp {
    fn default() -> Self {
        let mut app = Self {
            runs: Vec::new(),
            selected: 0,
            input_dtype: 0,
            output_dtype: 0,
            problem: ProblemKind::Matmul,
            shapes: vec![ProblemKind::Matmul.default_shape()],
            only_short_circuits: false,
            status: String::from("Ready."),
            output: LayoutJob::default(),
            ansi_style: AnsiStyle::default(),
            text_color: Color32::GRAY,
            run_rx: None,
            run_cancel: None,
            pending_run: None,
            rename_buffer: None,
            built_backends: Vec::new(),
            comparison_mode: false,
            remote: RemoteConfig::load(),
            conn_test: None,
            force_sync: false,
            disable_throughput_cache: false,
        };
        app.rescan_backends();
        app.rescan_runs(None);
        app
    }
}

impl AutotuneObservabilityApp {
    fn running(&self) -> bool {
        self.run_rx.is_some()
    }

    fn testing(&self) -> bool {
        self.conn_test.is_some()
    }

    fn poll_connection_test(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.conn_test else { return };
        match rx.try_recv() {
            Ok(result) => {
                self.status = result;
                self.conn_test = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint(),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.conn_test = None,
        }
    }
}

impl eframe::App for AutotuneObservabilityApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.text_color = ui.visuals().text_color();
        self.poll_run(&ctx);
        self.poll_connection_test(&ctx);

        self.render_controls_panel(ui);
        self.render_output_panel(ui);
        self.render_runs_panel(ui);
        self.render_events_panel(ui);
    }
}
