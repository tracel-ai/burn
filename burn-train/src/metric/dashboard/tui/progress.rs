use super::TerminalFrame;
use crate::metric::dashboard::TrainingProgress;
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
};
use std::time::Instant;

/// Simple progress bar for the training.
///
/// We currently ignore the time taken for the validation part.
pub(crate) struct ProgressBarState {
    progress_train: f64,
    started: Instant,
}

impl Default for ProgressBarState {
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

impl ProgressBarState {
    /// Update the training progress.
    pub(crate) fn update_train(&mut self, progress: &TrainingProgress) {
        let total_items = progress.progress.items_total * progress.epoch_total;
        let epoch_items = (progress.epoch - 1) * progress.progress.items_total;
        let iteration_items = progress.progress.items_processed as f64;

        self.progress_train = (epoch_items as f64 + iteration_items) / total_items as f64
    }

    /// Update the validation progress.
    pub(crate) fn update_valid(&mut self, _progress: &TrainingProgress) {
        // We don't use the validation for the progress yet.
    }

    /// Create a view for the current progress.
    pub(crate) fn view(&self) -> ProgressBarView {
        let eta = self.started.elapsed();
        let total_estimated = (eta.as_secs() as f64) / self.progress_train;

        let eta = if total_estimated.is_normal() {
            let remaining = 1.0 - self.progress_train;
            let eta = (total_estimated * remaining) as u64;
            format_eta(eta)
        } else {
            "---".to_string()
        };
        ProgressBarView::new(self.progress_train, eta)
    }
}

#[derive(new)]
pub(crate) struct ProgressBarView {
    progress: f64,
    eta: String,
}

impl ProgressBarView {
    /// Render the view.
    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
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
                    Constraint::Length(1), // Empty space
                    Constraint::Min(0),
                    Constraint::Length(self.eta.len() as u16 + 4),
                ]
                .as_ref(),
            )
            .split(size);

        let size_gauge = chunks[1];
        let size_eta = chunks[2];

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_eta() {
        assert_eq!("55 secs", format_eta(55), "Less than 1 minutes");
        assert_eq!("1 mins", format_eta(61), "More than 1 minutes");
        assert_eq!("2 mins", format_eta(2 * 61), "More than 2 minutes");
        assert_eq!("1 hours", format_eta(3601), "More than 1 hour");
        assert_eq!("2 hours", format_eta(2 * 3601), "More than 2 hour");
        assert_eq!("1 days", format_eta(24 * 3601), "More than 1 day");
        assert_eq!("2 days", format_eta(48 * 3601), "More than 2 day");
    }
}
