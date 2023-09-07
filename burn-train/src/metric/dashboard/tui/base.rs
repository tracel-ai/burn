use super::{ControlsView, NumericMetricView, ProgressView, TFrame, TextMetricView};
use ratatui::prelude::{Constraint, Direction, Layout, Rect};

#[derive(new)]
pub(crate) struct DashboardView<'a> {
    metric_numeric: NumericMetricView<'a>,
    metric_text: TextMetricView,
    progress: ProgressView<'a>,
    controls: ControlsView,
}

impl<'a> DashboardView<'a> {
    pub(crate) fn render<'b>(self, frame: &mut TFrame<'b>, size: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(16), Constraint::Max(3)].as_ref())
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
            .constraints([Constraint::Min(16), Constraint::Max(5)].as_ref())
            .split(size_other);
        let size_metric_text = chunks[0];
        let size_controls = chunks[1];

        self.metric_numeric.render(frame, size_metric_numeric);
        self.metric_text.render(frame, size_metric_text);
        self.controls.render(frame, size_controls);
        self.progress.render(frame, size_progress);
    }
}
