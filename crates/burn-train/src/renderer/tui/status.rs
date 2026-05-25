use crate::renderer::OverallProgress;

use super::TerminalFrame;
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

/// Show the training status with various information.
pub(crate) struct StatusState {
    progress: Option<OverallProgress>,
    mode: Mode,
}

enum Mode {
    Valid,
    Train,
    Evaluation,
}

impl Default for StatusState {
    fn default() -> Self {
        Self {
            progress: None,
            mode: Mode::Train,
        }
    }
}

impl StatusState {
    /// Update the training information.
    pub(crate) fn update_train(&mut self, progress: &OverallProgress) {
        self.progress = Some(progress.clone());
        self.mode = Mode::Train;
    }
    /// Update the validation information.
    pub(crate) fn update_valid(&mut self, progress: &OverallProgress) {
        self.progress = Some(progress.clone());
        self.mode = Mode::Valid;
    }
    /// Update the testing information.
    pub(crate) fn update_test(&mut self, progress: &OverallProgress) {
        self.progress = Some(progress.clone());
        self.mode = Mode::Evaluation;
    }
    /// Create a view.
    pub(crate) fn view(&self) -> StatusView {
        StatusView::new(self.progress.as_ref(), &self.mode)
    }
}

pub(crate) struct StatusView {
    lines: Vec<Vec<Span<'static>>>,
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

impl StatusView {
    fn new(progress: Option<&OverallProgress>, mode: &Mode) -> Self {
        let title = |title: &str| Span::from(format!(" {title} ")).bold().yellow();
        let value = |value: String| Span::from(value).italic();
        let mode_str = match mode {
            Mode::Valid => "Validating",
            Mode::Train => "Training",
            Mode::Evaluation => "Evaluation",
        };

        let width = progress
            .map(|p| {
                p.global_progress
                    .unit
                    .len()
                    .max(p.split_progress.unit.len())
            })
            .unwrap_or(0)
            .max("Mode".len());

        let mut lines = vec![vec![
            title(&format!("{: <width$} :", "Mode")),
            value(mode_str.to_string()),
        ]];

        if let Some(p) = progress {
            let g = &p.global_progress;
            let s = &p.split_progress;
            lines.push(vec![
                title(&format!("{: <width$} :", capitalize(&g.unit))),
                value(format!("{}/{}", g.items_processed, g.items_total)),
            ]);
            lines.push(vec![
                title(&format!("{: <width$} :", capitalize(&s.unit))),
                value(format!("{}/{}", s.items_processed, s.items_total)),
            ]);
        }

        Self { lines }
    }

    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        let paragraph = Paragraph::new(self.lines.into_iter().map(Line::from).collect::<Vec<_>>())
            .alignment(Alignment::Left)
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::Gray));

        frame.render_widget(paragraph, size);
    }
}
