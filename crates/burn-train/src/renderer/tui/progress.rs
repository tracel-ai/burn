use super::TerminalFrame;
use crate::{logger::ProgressSnapshot, renderer::tui::TuiSplit};
use burn_core::data::dataloader::Progress;
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
    progress_total: Option<f64>, // Progress for total execution.
    progress_task: Option<f64>,  // Progress for current task.
    task_label: String,
    show_eta: bool,
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
            progress_total: None,
            progress_task: None,
            task_label: String::new(),
            show_eta: false,
            split: TuiSplit::Train,
            estimate: ProgressEstimate::new(),
            starting_epoch: checkpoint.unwrap_or(0),
        }
    }

    /// Update the training progress.
    pub(crate) fn update_train(&mut self, progress: &ProgressSnapshot) {
        self.progress_total = calculate_progress(progress, 0, 0);
        self.progress_task = fraction(&progress.split);
        self.task_label = progress_label(&progress.split);
        self.show_eta = allows_eta(&progress.split);
        self.estimate.update(progress, self.starting_epoch);
        self.split = TuiSplit::Train;
    }

    /// Update the validation progress.
    pub(crate) fn update_valid(&mut self, progress: &ProgressSnapshot) {
        // We don't use the validation for the total progress yet.
        self.progress_task = fraction(&progress.split);
        self.task_label = progress_label(&progress.split);
        self.show_eta = allows_eta(&progress.split);
        self.split = TuiSplit::Valid;
    }

    /// Update the testing progress.
    pub(crate) fn update_test(&mut self, progress: &ProgressSnapshot) {
        // We don't use the testing for the total progress yet.
        self.progress_task = fraction(&progress.split);
        self.task_label = progress_label(&progress.split);
        self.show_eta = allows_eta(&progress.split);
        self.split = TuiSplit::Test;
    }

    /// Create a view for the current progress.
    pub(crate) fn view(&self) -> ProgressBarView {
        const NO_ETA: &str = "---";

        let eta = match self.show_eta.then(|| self.estimate.secs()).flatten() {
            Some(eta) => format_eta(eta),
            None => NO_ETA.to_string(),
        };
        ProgressBarView::new(
            self.progress_total,
            self.progress_task,
            self.task_label.clone(),
            self.split.color(),
            eta,
        )
    }
}

#[derive(new)]
pub(crate) struct ProgressBarView {
    progress: Option<f64>,
    progress_task: Option<f64>,
    task_label: String,
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
            .ratio(self.progress.unwrap_or(0.0).clamp(0.0, 1.0))
            .label(self.progress.map_or_else(
                || "—".to_string(),
                |ratio| format!("{:.0}%", ratio.clamp(0.0, 1.0) * 100.0),
            ));
        let progress_task = Gauge::default()
            .gauge_style(Style::default().fg(self.color_task))
            .ratio(self.progress_task.unwrap_or(0.0).clamp(0.0, 1.0))
            .label(self.task_label);

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
    progress: Option<f64>,
}

impl ProgressEstimate {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            started_after_warmup: None,
            warmup_num_items: 0,
            progress: None,
        }
    }

    fn secs(&self) -> Option<u64> {
        let eta = self.started_after_warmup?.elapsed();

        let progress = self.progress?;
        let total_estimated = (eta.as_secs() as f64) / progress;

        if total_estimated.is_normal() {
            let remaining = (1.0 - progress).max(0.0);
            let eta = (total_estimated * remaining) as u64;
            Some(eta)
        } else {
            None
        }
    }

    fn update(&mut self, progress: &ProgressSnapshot, starting_epoch: usize) {
        if !allows_eta(&progress.split) {
            self.progress = None;
            return;
        }
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
        if progress.split.items_processed >= WARMUP_NUM_ITERATION
            && self.started.elapsed() > Duration::from_secs(10)
        {
            self.init(progress, starting_epoch);
        }
    }

    fn init(&mut self, progress: &ProgressSnapshot, starting_epoch: usize) {
        let epoch = progress
            .global
            .items_processed
            .saturating_sub(starting_epoch);
        let local = &progress.split;

        let epoch_items = epoch.saturating_sub(1) * local.total.unwrap_or(0);
        let iteration_items = local.items_processed;
        self.warmup_num_items = epoch_items + iteration_items;

        self.started_after_warmup = Some(Instant::now());
        self.progress = calculate_progress(progress, starting_epoch, self.warmup_num_items);
    }
}

fn allows_eta(progress: &Progress) -> bool {
    progress.total.is_some()
}

fn fraction(progress: &Progress) -> Option<f64> {
    let total = progress.total?;
    Some(if total == 0 {
        0.0
    } else {
        (progress.items_processed as f64 / total as f64).clamp(0.0, 1.0)
    })
}

fn progress_label(progress: &Progress) -> String {
    match fraction(progress) {
        Some(ratio) => format!("{:.0}%", ratio * 100.0),
        None => format!(
            "{} {}",
            progress.items_processed,
            progress.unit.as_deref().unwrap_or("items")
        ),
    }
}

fn calculate_progress(
    progress: &ProgressSnapshot,
    starting_epoch: usize,
    ignore_num_items: usize,
) -> Option<f64> {
    // Step-based runs have no epoch dimension.
    if progress.global.unit.as_deref() != Some("epochs") {
        return fraction(&progress.split);
    }
    let epoch_total = progress.global.total?.saturating_sub(starting_epoch);
    let epoch = progress
        .global
        .items_processed
        .saturating_sub(starting_epoch);
    let total = progress.split.total?;
    if total == 0 {
        return Some(if epoch_total == 0 {
            0.0
        } else {
            (epoch.saturating_sub(1) as f64) / epoch_total as f64
        });
    }
    let total_items = total * epoch_total;
    let epoch_items = epoch.saturating_sub(1) * total;
    let iteration_items = progress.split.items_processed;
    let num_items = (epoch_items + iteration_items).saturating_sub(ignore_num_items);
    Some(if total_items == 0 {
        0.0
    } else {
        (num_items as f64 / total_items as f64).clamp(0.0, 1.0)
    })
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
        let progress = ProgressSnapshot::new(
            Progress::new(9, 10, Some("epochs".to_string())),
            Progress::new(5, 10, Some("items".to_string())),
        );

        let starting_epoch = 8;
        let result = calculate_progress(&progress, starting_epoch, 0);

        // Two epochs remaining while the first is half done.
        assert_eq!(Some(0.25), result);
    }

    #[test]
    fn calculate_progress_for_eta_with_warmup() {
        let progress = ProgressSnapshot::new(
            Progress::new(9, 10, Some("epochs".to_string())),
            Progress::new(110, 1000, Some("items".to_string())),
        );

        let starting_epoch = 8;
        let result = calculate_progress(&progress, starting_epoch, 10);

        // Two epochs remaining while the first is 11% done, minus 10 warmup items.
        assert_eq!(Some(0.05), result);
    }
    #[test]
    fn unknown_progress_controls_label_ratio_and_eta() {
        let mut state = ProgressBarState::new(None);
        state.estimate.started_after_warmup = Some(Instant::now() - Duration::from_secs(10));
        let progress = ProgressSnapshot::new(
            Progress::new(1, 2, Some("epochs".into())),
            Progress::new(6, None, None),
        );
        state.update_train(&progress);
        assert_eq!(state.progress_task, None);
        assert_eq!(state.task_label, "6 items");
        assert_eq!(state.view().eta, "---");
        assert_eq!(state.estimate.secs(), None);
        let progress = ProgressSnapshot::new(
            Progress::new(1, 2, Some("epochs".into())),
            Progress::new(2, 10, None),
        );
        state.update_train(&progress);
        assert!(state.estimate.secs().is_some());
    }
}
