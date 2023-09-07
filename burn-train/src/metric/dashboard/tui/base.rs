use super::{NumericMetricView, ProgressView, TFrame, TextMetricView};
use ratatui::prelude::{Constraint, Direction, Layout, Rect};

#[derive(new)]
pub(crate) struct DashboardView<'a> {
    metric_numeric: NumericMetricView<'a>,
    metric_text: TextMetricView,
    progress: ProgressView<'a>,
}

impl<'a> DashboardView<'a> {
    pub(crate) fn render<'b>(self, frame: &mut TFrame<'b>, size: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(16), Constraint::Max(3)].as_ref())
            .split(size);
        let size_metrics = chunks[0];
        let size_progress = chunks[1];

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(size_metrics);
        let size_metric_text = chunks[0];
        let size_metric_numeric = chunks[1];

        self.metric_numeric.render(frame, size_metric_numeric);
        self.metric_text.render(frame, size_metric_text);
        self.progress.render(frame, size_progress);
    }
}
