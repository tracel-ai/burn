use crate::renderer::TrainingProgress;

use super::TerminalFrame;
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
    progress_train: f64,         // Progress for total training.
    progress_train_for_eta: f64, // Progress considering the starting epoch.
    starting_epoch: usize,
    started: Instant,
}

const MINUTE: u64 = 60;
const HOUR: u64 = 60 * 60;
const DAY: u64 = 24 * 60 * 60;

impl ProgressBarState {
    pub fn new(checkpoint: Option<usize>) -> Self {
        Self {
            progress_train: 0.0,
            progress_train_for_eta: 0.0,
            started: Instant::now(),
            starting_epoch: checkpoint.unwrap_or(0),
        }
    }
    /// Update the training progress.
    pub(crate) fn update_train(&mut self, progress: &TrainingProgress) {
        self.progress_train = calculate_progress(progress, 0);
        self.progress_train_for_eta = calculate_progress(progress, self.starting_epoch);
    }

    /// Update the validation progress.
    pub(crate) fn update_valid(&mut self, _progress: &TrainingProgress) {
        // We don't use the validation for the progress yet.
    }

    /// Create a view for the current progress.
    pub(crate) fn view(&self) -> ProgressBarView {
        let eta = self.started.elapsed();
        let total_estimated = (eta.as_secs() as f64) / self.progress_train_for_eta;

        let eta = if total_estimated.is_normal() {
            let remaining = 1.0 - self.progress_train_for_eta;
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

fn calculate_progress(progress: &TrainingProgress, starting_epoch: usize) -> f64 {
    let epoch_total = progress.epoch_total - starting_epoch;
    let epoch = progress.epoch - starting_epoch;

    let total_items = progress.progress.items_total * epoch_total;
    let epoch_items = (epoch - 1) * progress.progress.items_total;
    let iteration_items = progress.progress.items_processed as f64;

    (epoch_items as f64 + iteration_items) / total_items as f64
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
    use burn_core::data::dataloader::Progress;

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

    #[test]
    fn calculate_progress_for_eta() {
        let half = Progress {
            items_processed: 5,
            items_total: 10,
        };
        let progress = TrainingProgress {
            progress: half,
            epoch: 9,
            epoch_total: 10,
            iteration: 500,
        };

        let starting_epoch = 8;
        let progress = calculate_progress(&progress, starting_epoch);

        // Two epochs remaining while the first is half done.
        assert_eq!(0.25, progress);
    }
}
