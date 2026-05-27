use std::collections::BTreeMap;

use crate::logger::ProgressSnapshot;

use super::TerminalFrame;
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

/// Show the training status with various information.
pub(crate) struct StatusState {
    progress: Option<ProgressSnapshot>,
    mode: Mode,
    event_counters: BTreeMap<String, usize>,
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
            event_counters: BTreeMap::new(),
        }
    }
}

impl StatusState {
    /// Update the training information.
    pub(crate) fn update_train(&mut self, progress: &ProgressSnapshot) {
        self.progress = Some(progress.clone());
        self.mode = Mode::Train;
    }
    /// Update the validation information.
    pub(crate) fn update_valid(&mut self, progress: &ProgressSnapshot) {
        self.progress = Some(progress.clone());
        self.mode = Mode::Valid;
    }
    /// Update the testing information.
    pub(crate) fn update_test(&mut self, progress: &ProgressSnapshot) {
        self.progress = Some(progress.clone());
        self.mode = Mode::Evaluation;
    }
    /// Update counters from a progress event.
    pub(crate) fn update_counter(&mut self, event: String) {
        *self.event_counters.entry(event).or_insert(0) += 1;
    }

    /// Reset all counters at the end of a split.
    pub(crate) fn reset_counters(&mut self) {
        for val in self.event_counters.values_mut() {
            *val = 0;
        }
    }

    /// Create a view.
    pub(crate) fn view(&self) -> StatusView {
        StatusView::new(self.progress.as_ref(), &self.mode, &self.event_counters)
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
    fn new(
        progress: Option<&ProgressSnapshot>,
        mode: &Mode,
        event_counters: &BTreeMap<String, usize>,
    ) -> Self {
        let title = |title: &str| Span::from(format!(" {title} ")).bold().yellow();
        let value = |value: String| Span::from(value).italic();
        let mode_str = match mode {
            Mode::Valid => "Validating",
            Mode::Train => "Training",
            Mode::Evaluation => "Evaluation",
        };

        let width = progress
            .map(|p| {
                p.global
                    .unit
                    .as_deref()
                    .map_or(0, |s| s.len())
                    .max(p.split.unit.as_deref().map_or(0, |s| s.len()))
            })
            .unwrap_or(0)
            .max("Mode".len())
            .max(event_counters.keys().map(|k| k.len()).max().unwrap_or(0));

        let mut lines = vec![vec![
            title(&format!("{: <width$} :", "Mode")),
            value(mode_str.to_string()),
        ]];

        if let Some(p) = progress {
            let g = &p.global;
            lines.push(vec![
                title(&format!(
                    "{: <width$} :",
                    capitalize(g.unit.as_deref().unwrap_or(""))
                )),
                value(format!("{}/{}", g.items_processed, g.items_total)),
            ]);
            let s = &p.split;
            lines.push(vec![
                title(&format!(
                    "{: <width$} :",
                    capitalize(s.unit.as_deref().unwrap_or(""))
                )),
                value(format!("{}/{}", s.items_processed, s.items_total)),
            ]);
        }

        for (key, val) in event_counters {
            lines.push(vec![
                title(&format!("{: <width$} :", key)),
                value(format!("{val}")),
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
