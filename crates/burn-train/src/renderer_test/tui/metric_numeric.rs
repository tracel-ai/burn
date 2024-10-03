use crate::renderer::TrainingProgress;

use super::{FullHistoryPlot, RecentHistoryPlot, TerminalFrame};
use crossterm::event::{Event, KeyCode, KeyEventKind};
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Paragraph, Tabs},
};
use std::collections::HashMap;

/// 1000 seems to be required to see some improvement.
const MAX_NUM_SAMPLES_RECENT: usize = 1000;
/// 250 seems to be the right resolution when plotting all history.
/// Otherwise, there is too much points and the lines arent't smooth enough.
const MAX_NUM_SAMPLES_FULL: usize = 250;

/// Numeric metrics state that handles creating plots.
#[derive(Default)]
pub(crate) struct NumericMetricsState {
    data: HashMap<String, (RecentHistoryPlot, FullHistoryPlot)>,
    names: Vec<String>,
    selected: usize,
    kind: PlotKind,
    num_samples_train: Option<usize>,
    num_samples_valid: Option<usize>,
}

/// The kind of plot to display.
#[derive(Default, Clone, Copy)]
pub(crate) enum PlotKind {
    /// Display the full history of the metric with reduced resolution.
    #[default]
    Full,
    /// Display only the recent history of the metric, but with more resolution.
    Recent,
}

impl NumericMetricsState {
    /// Register a new training value for the metric with the given name.
    pub(crate) fn push_train(&mut self, name: String, data: f64) {
        if let Some((recent, full)) = self.data.get_mut(&name) {
            recent.push_train(data);
            full.push_train(data);
        } else {
            let mut recent = RecentHistoryPlot::new(MAX_NUM_SAMPLES_RECENT);
            let mut full = FullHistoryPlot::new(MAX_NUM_SAMPLES_FULL);

            recent.push_train(data);
            full.push_train(data);

            self.names.push(name.clone());
            self.data.insert(name, (recent, full));
        }
    }

    /// Register a new validation value for the metric with the given name.
    pub(crate) fn push_valid(&mut self, key: String, data: f64) {
        if let Some((recent, full)) = self.data.get_mut(&key) {
            recent.push_valid(data);
            full.push_valid(data);
        } else {
            let mut recent = RecentHistoryPlot::new(MAX_NUM_SAMPLES_RECENT);
            let mut full = FullHistoryPlot::new(MAX_NUM_SAMPLES_FULL);

            recent.push_valid(data);
            full.push_valid(data);

            self.data.insert(key, (recent, full));
        }
    }

    /// Update the state with the training progress.
    pub(crate) fn update_progress_train(&mut self, progress: &TrainingProgress) {
        if self.num_samples_train.is_some() {
            return;
        }

        self.num_samples_train = Some(progress.progress.items_total);
    }

    /// Update the state with the validation progress.
    pub(crate) fn update_progress_valid(&mut self, progress: &TrainingProgress) {
        if self.num_samples_valid.is_some() {
            return;
        }

        if let Some(num_sample_train) = self.num_samples_train {
            for (_, (_recent, full)) in self.data.iter_mut() {
                let ratio = progress.progress.items_total as f64 / num_sample_train as f64;
                full.update_max_sample_valid(ratio);
            }
        }

        self.num_samples_valid = Some(progress.progress.items_total);
    }

    /// Create a view to display the numeric metrics.
    pub(crate) fn view(&self) -> NumericMetricView<'_> {
        match self.names.is_empty() {
            true => NumericMetricView::None,
            false => NumericMetricView::Plots(&self.names, self.selected, self.chart(), self.kind),
        }
    }

    /// Handle the current event.
    pub(crate) fn on_event(&mut self, event: &Event) {
        if let Event::Key(key) = event {
            match key.kind {
                KeyEventKind::Release | KeyEventKind::Repeat => (),
                #[cfg(target_os = "windows")] // Fix the double toggle on Windows.
                KeyEventKind::Press => return,
                #[cfg(not(target_os = "windows"))]
                KeyEventKind::Press => (),
            }
            match key.code {
                KeyCode::Right => self.next_metric(),
                KeyCode::Left => self.previous_metric(),
                KeyCode::Up => self.switch_kind(),
                KeyCode::Down => self.switch_kind(),
                _ => {}
            }
        }
    }

    fn switch_kind(&mut self) {
        self.kind = match self.kind {
            PlotKind::Full => PlotKind::Recent,
            PlotKind::Recent => PlotKind::Full,
        };
    }

    fn next_metric(&mut self) {
        self.selected = (self.selected + 1) % {
            let this = &self;
            this.data.len()
        };
    }

    fn previous_metric(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        } else {
            self.selected = ({
                let this = &self;
                this.data.len()
            }) - 1;
        }
    }

    fn chart<'a>(&'a self) -> Chart<'a> {
        let name = self.names.get(self.selected).unwrap();
        let (recent, full) = self.data.get(name).unwrap();

        let (datasets, axes) = match self.kind {
            PlotKind::Full => (full.datasets(), &full.axes),
            PlotKind::Recent => (recent.datasets(), &recent.axes),
        };

        Chart::<'a>::new(datasets)
            .block(Block::default())
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .title("Iteration")
                    .labels(
                        axes.labels_x
                            .clone()
                            .into_iter()
                            .map(|s| s.bold())
                            .collect(),
                    )
                    .bounds(axes.bounds_x),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .labels(
                        axes.labels_y
                            .clone()
                            .into_iter()
                            .map(|s| s.bold())
                            .collect(),
                    )
                    .bounds(axes.bounds_y),
            )
    }
}

#[derive(new)]
pub(crate) enum NumericMetricView<'a> {
    Plots(&'a [String], usize, Chart<'a>, PlotKind),
    None,
}

impl<'a> NumericMetricView<'a> {
    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        match self {
            Self::Plots(titles, selected, chart, kind) => {
                let block = Block::default()
                    .borders(Borders::ALL)
                    .title("Plots")
                    .title_alignment(Alignment::Left);
                let size_new = block.inner(size);
                frame.render_widget(block, size);

                let size = size_new;

                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(
                        [
                            Constraint::Length(2),
                            Constraint::Length(1),
                            Constraint::Min(0),
                        ]
                        .as_ref(),
                    )
                    .split(size);

                let tabs = Tabs::new(titles.iter().map(|i| Line::from(vec![i.clone().yellow()])))
                    .select(selected)
                    .style(Style::default())
                    .highlight_style(
                        Style::default()
                            .add_modifier(Modifier::BOLD)
                            .add_modifier(Modifier::UNDERLINED)
                            .fg(Color::LightYellow),
                    );
                let title = match kind {
                    PlotKind::Full => "Full History",
                    PlotKind::Recent => "Recent History",
                };

                let plot_type =
                    Paragraph::new(Line::from(title.bold())).alignment(Alignment::Center);

                frame.render_widget(tabs, chunks[0]);
                frame.render_widget(plot_type, chunks[1]);
                frame.render_widget(chart, chunks[2]);
            }
            Self::None => {}
        };
    }
}
