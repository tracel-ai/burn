use crate::metric::dashboard::TrainingProgress;

use super::{FullHistoryPlot, RecentHistoryPlot, TFrame};
use crossterm::event::{Event, KeyCode};
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Paragraph, Tabs},
};
use std::collections::HashMap;

static MAX_NUM_SAMPLES_RECENT: usize = 1000;
static MAX_NUM_SAMPLES_FULL: usize = 250;

#[derive(Default)]
pub(crate) struct NumericMetricsState {
    data: HashMap<String, (RecentHistoryPlot, FullHistoryPlot)>,
    names: Vec<String>,
    selected: usize,
    kind: PlotKind,
    num_samples_train: Option<usize>,
    num_samples_valid: Option<usize>,
}

#[derive(Default, Clone, Copy)]
pub(crate) enum PlotKind {
    #[default]
    Full,
    Recent,
}

impl NumericMetricsState {
    pub(crate) fn push_train(&mut self, key: String, data: f64) {
        if let Some((recent, full)) = self.data.get_mut(&key) {
            recent.push_train(data);
            full.push_train(data);
        } else {
            let mut recent = RecentHistoryPlot::new(MAX_NUM_SAMPLES_RECENT);
            let mut full = FullHistoryPlot::new(MAX_NUM_SAMPLES_FULL);

            recent.push_train(data);
            full.push_train(data);

            self.names.push(key.clone());
            self.data.insert(key, (recent, full));
        }
    }

    pub(crate) fn update_progress_train(&mut self, progress: &TrainingProgress) {
        if self.num_samples_train.is_some() {
            return;
        }

        self.num_samples_train = Some(progress.progress.items_total);
    }

    pub(crate) fn update_progress_valid(&mut self, progress: &TrainingProgress) {
        if self.num_samples_valid.is_some() {
            return;
        }

        if let Some(num_sample_train) = self.num_samples_train {
            for (_, (_recent, full)) in self.data.iter_mut() {
                let max_samples =
                    progress.progress.items_total * MAX_NUM_SAMPLES_FULL / num_sample_train;
                full.update_max_sample_valid(max_samples);
            }
        }

        self.num_samples_valid = Some(progress.progress.items_total);
    }

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

    pub(crate) fn view<'a>(&'a self) -> NumericMetricView<'a> {
        match self.names.is_empty() {
            true => NumericMetricView::None,
            false => NumericMetricView::Plots(&self.names, self.selected, self.chart(), self.kind),
        }
    }

    pub(crate) fn on_event(&mut self, event: &Event) {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Right => self.next(),
                KeyCode::Left => self.previous(),
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

    fn next(&mut self) {
        self.selected = (self.selected + 1) % {
            let ref this = self;
            this.data.len()
        };
    }

    fn previous(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        } else {
            self.selected = ({
                let ref this = self;
                this.data.len()
            }) - 1;
        }
    }

    fn chart<'a>(&'a self) -> Chart<'a> {
        let name = self.names.get(self.selected).unwrap();
        let (recent, full) = self.data.get(name).unwrap();

        let (datasets, labels_x, labels_y, bounds_x, bounds_y) = match self.kind {
            PlotKind::Full => (
                full.datasets(),
                &full.labels_x,
                &full.labels_y,
                full.bounds_x,
                full.bounds_y,
            ),
            PlotKind::Recent => (
                recent.datasets(),
                &recent.labels_x,
                &recent.labels_y,
                recent.bounds_x,
                recent.bounds_y,
            ),
        };

        Chart::<'a>::new(datasets)
            .block(Block::default())
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .title("Iteration")
                    .labels(labels_x.iter().map(|s| s.bold()).collect())
                    .bounds(bounds_x),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .labels(labels_y.iter().map(|s| s.bold()).collect())
                    .bounds(bounds_y),
            )
    }
}

#[derive(new)]
pub(crate) enum NumericMetricView<'a> {
    Plots(&'a [String], usize, Chart<'a>, PlotKind),
    None,
}

impl<'a> NumericMetricView<'a> {
    pub(crate) fn render<'b>(self, frame: &mut TFrame<'b>, size: Rect) {
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

                let titles = titles
                    .iter()
                    .map(|i| Line::from(vec![i.yellow()]))
                    .collect();

                let tabs = Tabs::new(titles)
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
