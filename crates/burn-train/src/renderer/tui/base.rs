use std::sync::Arc;

use super::{
    ControlsView, NumericMetricView, ProgressBarView, StatusView, TerminalFrame, TextMetricView,
};
use ratatui::{
    prelude::{Constraint, Direction, Layout, Rect},
    style::Color,
};

#[derive(new)]
pub(crate) struct MetricsView<'a> {
    metric_numeric: NumericMetricView<'a>,
    metric_text: TextMetricView,
    progress: ProgressBarView,
    controls: ControlsView,
    status: StatusView,
}

impl MetricsView<'_> {
    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(16), Constraint::Max(4)].as_ref())
            .split(size);
        let size_other = chunks[0];
        let size_progress = chunks[1];

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(38), Constraint::Percentage(62)].as_ref())
            .split(size_other);
        let size_other = chunks[0];
        let size_metric_numeric = chunks[1];

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Max(5), Constraint::Min(6), Constraint::Max(6)].as_ref())
            .split(size_other);
        let size_controls = chunks[0];
        let size_metric_text = chunks[1];
        let size_status = chunks[2];

        self.metric_numeric.render(frame, size_metric_numeric);
        self.metric_text.render(frame, size_metric_text);
        self.controls.render(frame, size_controls);
        self.progress.render(frame, size_progress);
        self.status.render(frame, size_status);
    }
}

#[derive(Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TuiSplit {
    Train,
    Valid,
    Test,
}

#[derive(Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TuiGroup {
    Default,
    Named(Arc<String>),
}

#[derive(new, Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TuiTag {
    pub(crate) split: TuiSplit,
    pub(crate) group: TuiGroup,
}

impl core::fmt::Display for TuiTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.group {
            TuiGroup::Default => f.write_fmt(format_args!("{}", self.split)),
            TuiGroup::Named(group) => f.write_fmt(format_args!("{} - {}", self.split, group)),
        }
    }
}
impl core::fmt::Display for TuiGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuiGroup::Default => f.write_str(""),
            TuiGroup::Named(group) => f.write_fmt(format_args!("{group} ")),
        }
    }
}

impl core::fmt::Display for TuiSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuiSplit::Train => f.write_str("Train"),
            TuiSplit::Valid => f.write_str("Valid"),
            TuiSplit::Test => f.write_str("Test"),
        }
    }
}

impl TuiSplit {
    pub(crate) fn color(&self) -> Color {
        match self {
            TuiSplit::Train => Color::LightRed,
            TuiSplit::Valid => Color::LightBlue,
            TuiSplit::Test => Color::LightGreen,
        }
    }
}
