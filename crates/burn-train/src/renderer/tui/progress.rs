use super::TerminalFrame;
use crate::renderer::{EvaluationProgress, TrainingProgress, tui::TuiSplit};
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
};
use std::time::{Duration, Instant};

/// Simple progress bar for the training.
///
/// We currently ignore the time taken for the validation part.
pub(crate) struct ProgressBarState {
    progress_total: f64, // Progress for total execution.
    progress_task: f64,  // Progress for current task.
    split: TuiSplit,
    starting_epoch: usize,
    estimate: ProgressEstimate,
}

const MINUTE: u64 = 60;
const HOUR: u64 = 60 * 60;
const DAY: u64 = 24 * 60 * 60;

impl ProgressBarState {
    pub fn new(checkpoint: Option<usize>) -> Self {
        Self {
            progress_total: 0.0,
            progress_task: 0.0,
            split: TuiSplit::Train,
            estimate: ProgressEstimate::new(),
            starting_epoch: checkpoint.unwrap_or(0),
        }
    }
    /// Update the training progress.
    pub(crate) fn update_train(&mut self, progress: &TrainingProgress) {
        self.progress_total = calculate_progress(progress, 0, 0);
        self.progress_task =
            progress.progress.items_processed as f64 / progress.progress.items_total as f64;
        self.estimate.update(progress, self.starting_epoch);
        self.split = TuiSplit::Train;
    }

    /// Update the validation progress.
    pub(crate) fn update_valid(&mut self, progress: &TrainingProgress) {
        // We don't use the validation for the total progress yet.
        self.progress_task =
            progress.progress.items_processed as f64 / progress.progress.items_total as f64;
        self.split = TuiSplit::Valid;
    }

    /// Update the testing progress.
    pub(crate) fn update_test(&mut self, progress: &EvaluationProgress) {
        // We don't use the testing for the total progress yet.
        self.progress_task =
            progress.progress.items_processed as f64 / progress.progress.items_total as f64;
        self.split = TuiSplit::Test;
    }

    /// Create a view for the current progress.
    pub(crate) fn view(&self) -> ProgressBarView {
        const NO_ETA: &str = "---";

        let eta = match self.estimate.secs() {
            Some(eta) => format_eta(eta),
            None => NO_ETA.to_string(),
        };
        ProgressBarView::new(
            self.progress_total,
            self.progress_task,
            self.split.color(),
            eta,
        )
    }
}

#[derive(new)]
pub(crate) struct ProgressBarView {
    progress: f64,
    progress_task: f64,
    color_task: Color,
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
            .direction(Direction::Vertical)
            .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)].as_ref())
            .split(size);

        let size_task = chunks[0];
        let size_total = chunks[1];

        let calculate_size = |size: Rect| {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints(
                    [
                        Constraint::Length(1), // Empty space
                        Constraint::Min(0),
                        Constraint::Length(self.eta.len() as u16 + 4),
                    ]
                    .as_ref(),
                )
                .split(size)
        };

        let chunks = calculate_size(size_total);
        let size_gauge_total = chunks[1];
        let size_eta = chunks[2];
        let chunks = calculate_size(size_task);
        let size_gauge_task = chunks[1];

        let progress_total = Gauge::default()
            .gauge_style(Style::default().fg(Color::Yellow))
            .ratio(self.progress.min(1.0));
        let progress_task = Gauge::default()
            .gauge_style(Style::default().fg(self.color_task))
            .ratio(self.progress_task.min(1.0));

        let eta = Paragraph::new(Line::from(vec![
            Span::from(" ("),
            Span::from(self.eta).italic(),
            Span::from(") "),
        ]));

        frame.render_widget(progress_task, size_gauge_task);
        frame.render_widget(progress_total, size_gauge_total);
        frame.render_widget(eta, size_eta);
    }
}

struct ProgressEstimate {
    started: Instant,
    started_after_warmup: Option<Instant>,
    warmup_num_items: usize,
    progress: f64,
}

impl ProgressEstimate {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            started_after_warmup: None,
            warmup_num_items: 0,
            progress: 0.0,
        }
    }

    fn secs(&self) -> Option<u64> {
        let eta = self.started_after_warmup?.elapsed();

        let total_estimated = (eta.as_secs() as f64) / self.progress;

        if total_estimated.is_normal() {
            let remaining = 1.0 - self.progress;
            let eta = (total_estimated * remaining) as u64;
            Some(eta)
        } else {
            None
        }
    }

    fn update(&mut self, progress: &TrainingProgress, starting_epoch: usize) {
        if self.started_after_warmup.is_some() {
            self.progress = calculate_progress(progress, starting_epoch, self.warmup_num_items);
            return;
        }

        const WARMUP_NUM_ITERATION: usize = 10;

        // When the training has started since 30 seconds.
        if self.started.elapsed() > Duration::from_secs(30) {
            self.init(progress, starting_epoch);
            return;
        }

        // When the training has started since at least 10 seconds and completed 10 iterations.
        if progress.iteration >= WARMUP_NUM_ITERATION
            && self.started.elapsed() > Duration::from_secs(10)
        {
            self.init(progress, starting_epoch);
        }
    }

    fn init(&mut self, progress: &TrainingProgress, starting_epoch: usize) {
        let epoch = progress.epoch - starting_epoch;
        let epoch_items = (epoch - 1) * progress.progress.items_total;
        let iteration_items = progress.progress.items_processed;

        self.warmup_num_items = epoch_items + iteration_items;
        self.started_after_warmup = Some(Instant::now());
        self.progress = calculate_progress(progress, starting_epoch, self.warmup_num_items);
    }
}

fn calculate_progress(
    progress: &TrainingProgress,
    starting_epoch: usize,
    ignore_num_items: usize,
) -> f64 {
    let epoch_total = progress.epoch_total - starting_epoch;
    let epoch = progress.epoch - starting_epoch;

    let total_items = progress.progress.items_total * epoch_total;
    let epoch_items = (epoch - 1) * progress.progress.items_total;
    let iteration_items = progress.progress.items_processed;
    let num_items = epoch_items + iteration_items - ignore_num_items;

    num_items as f64 / total_items as f64
}

fn format_eta(eta_secs: u64) -> String {
    let seconds = eta_secs % 60;
    let minutes = eta_secs / MINUTE % 60;
    let hours = eta_secs / HOUR % 24;
    let days = eta_secs / DAY;

    if days > 1 {
        format!("{days} days")
    } else if days == 1 {
        "1 day".to_string()
    } else if hours > 1 {
        format!("{hours} hours")
    } else if hours == 1 {
        "1 hour".to_string()
    } else if minutes > 1 {
        format!("{minutes} mins")
    } else if minutes == 1 {
        "1 min".to_string()
    } else if seconds > 1 {
        format!("{seconds} secs")
    } else {
        "1 sec".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::data::dataloader::Progress;

    #[test]
    fn test_format_eta() {
        assert_eq!("55 secs", format_eta(55), "Less than 1 minutes");
        assert_eq!("1 min", format_eta(61), "More than 1 minutes");
        assert_eq!("2 mins", format_eta(2 * 61), "More than 2 minutes");
        assert_eq!("1 hour", format_eta(3601), "More than 1 hour");
        assert_eq!("2 hours", format_eta(2 * 3601), "More than 2 hour");
        assert_eq!("1 day", format_eta(24 * 3601), "More than 1 day");
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
        let progress = calculate_progress(&progress, starting_epoch, 0);

        // Two epochs remaining while the first is half done.
        assert_eq!(0.25, progress);
    }

    #[test]
    fn calculate_progress_for_eta_with_warmup() {
        let half = Progress {
            items_processed: 110,
            items_total: 1000,
        };
        let progress = TrainingProgress {
            progress: half,
            epoch: 9,
            epoch_total: 10,
            iteration: 500,
        };

        let starting_epoch = 8;
        let progress = calculate_progress(&progress, starting_epoch, 10);

        // Two epochs remaining while the first is half done.
        assert_eq!(0.05, progress);
    }
}
