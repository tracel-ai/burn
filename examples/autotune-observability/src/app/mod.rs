use std::sync::mpsc::Receiver;

use eframe::egui;
use egui::Color32;
use egui::text::LayoutJob;

use crate::ansi::AnsiStyle;
use crate::run_support::{MatmulShape, RunMsg, RunView};

mod actions;
mod panels;

pub struct AutotuneObservabilityApp {
    runs: Vec<RunView>,
    selected: usize,
    input_dtype: usize,
    output_dtype: usize,
    matmuls: Vec<MatmulShape>,
    only_short_circuits: bool,
    status: String,
    output: LayoutJob,
    ansi_style: AnsiStyle,
    text_color: Color32,
    run_rx: Option<Receiver<RunMsg>>,
    /// Directory name of the in-flight run, selected once it finishes successfully.
    pending_run: Option<String>,
}

impl Default for AutotuneObservabilityApp {
    fn default() -> Self {
        let mut app = Self {
            runs: Vec::new(),
            selected: 0,
            input_dtype: 0,
            output_dtype: 0,
            matmuls: vec![MatmulShape {
                m: 512,
                k: 512,
                n: 512,
            }],
            only_short_circuits: false,
            status: String::from("Ready."),
            output: LayoutJob::default(),
            ansi_style: AnsiStyle::default(),
            text_color: Color32::GRAY,
            run_rx: None,
            pending_run: None,
        };
        app.rescan_runs(None);
        app
    }
}

impl AutotuneObservabilityApp {
    fn running(&self) -> bool {
        self.run_rx.is_some()
    }
}

impl eframe::App for AutotuneObservabilityApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.text_color = ui.visuals().text_color();
        self.poll_run(&ctx);

        self.render_controls_panel(ui);
        self.render_output_panel(ui);
        self.render_runs_panel(ui);
        self.render_events_panel(ui);
    }
}
