use super::TFrame;
use crate::metric::dashboard::TrainingProgress;
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
};
use std::time::Instant;

pub(crate) struct ProgressState {
    progress_train: f64,
    started: Instant,
}

impl Default for ProgressState {
    fn default() -> Self {
        Self {
            progress_train: 0.0,
            started: Instant::now(),
        }
    }
}
const MINUTE: u64 = 60;
const HOUR: u64 = 60 * 60;
const DAY: u64 = 24 * 60 * 60;

impl ProgressState {
    pub(crate) fn update_train(&mut self, progress: &TrainingProgress) {
        let total_items = progress.progress.items_total * progress.epoch_total;
        let epoch_items = (progress.epoch - 1) * progress.progress.items_total;
        let iteration_items = progress.progress.items_processed as f64;

        self.progress_train = (epoch_items as f64 + iteration_items) / total_items as f64
    }

    pub(crate) fn update_valid(&mut self, _progress: &TrainingProgress) {
        // We don't use the validation for the progress yet.
    }

    pub(crate) fn view(&self) -> ProgressView {
        let eta = self.started.elapsed();
        let total_estimated = (eta.as_secs() as f64) / self.progress_train;

        let eta = if total_estimated.is_normal() {
            let remaining = 1.0 - self.progress_train;
            let eta = (total_estimated * remaining) as u64;
            format_eta(eta)
        } else {
            format!("---")
        };
        ProgressView::new(self.progress_train, eta)
    }
}

fn format_eta(eta_secs: u64) -> String {
    let seconds = eta_secs % 60;
    let minutes = eta_secs / MINUTE % 60;
    let hours = eta_secs / HOUR % 24;
    let days = eta_secs / DAY;

    if days > 0 {
        return format!("{days} days");
    }

    if hours > 0 {
        return format!("{hours} hours");
    }

    if minutes > 0 {
        return format!("{minutes} mins");
    }

    format!("{seconds} secs")
}

#[derive(new)]
pub(crate) struct ProgressView {
    progress: f64,
    eta: String,
}

impl ProgressView {
    pub(crate) fn render<'b>(self, frame: &mut TFrame<'b>, size: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Progress")
            .title_alignment(Alignment::Left);
        let size_new = block.inner(size);
        frame.render_widget(block, size);
        let size = size_new;

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Min(0),
                    Constraint::Length(self.eta.len() as u16 + 4),
                ]
                .as_ref(),
            )
            .split(size);

        let size_gauge = chunks[0];
        let size_eta = chunks[1];

        let iteration = Gauge::default()
            .gauge_style(Style::default().fg(Color::Yellow))
            .ratio(self.progress);
        let eta = Paragraph::new(Line::from(vec![
            Span::from(" ("),
            Span::from(self.eta).italic(),
            Span::from(") "),
        ]));

        frame.render_widget(iteration, size_gauge);
        frame.render_widget(eta, size_eta);
    }
}
