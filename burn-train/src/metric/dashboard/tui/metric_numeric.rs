use super::{RecentHistoryPlot, TFrame};
use crossterm::event::{Event, KeyCode};
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Paragraph, Tabs},
};
use std::collections::HashMap;

static MAX_NUM_SAMPLES: usize = 1000;

#[derive(Default)]
pub(crate) struct NumericMetricsState {
    data: HashMap<String, RecentHistoryPlot>,
    names: Vec<String>,
    selected: usize,
}

impl NumericMetricsState {
    pub(crate) fn push_train(&mut self, key: String, data: f64) {
        if let Some(existing) = self.data.get_mut(&key) {
            existing.push_train(data);
        } else {
            let mut chart = RecentHistoryPlot::new(MAX_NUM_SAMPLES);
            chart.push_train(data);
            self.names.push(key.clone());
            self.data.insert(key, chart);
        }
    }

    pub(crate) fn push_valid(&mut self, key: String, data: f64) {
        if let Some(existing) = self.data.get_mut(&key) {
            existing.push_valid(data);
        } else {
            let mut chart = RecentHistoryPlot::new(MAX_NUM_SAMPLES);
            chart.push_valid(data);
            self.data.insert(key, chart);
        }
    }

    pub(crate) fn view<'a>(&'a self) -> NumericMetricView<'a> {
        match self.names.is_empty() {
            true => NumericMetricView::None,
            false => NumericMetricView::Plots(&self.names, self.selected, self.chart()),
        }
    }

    pub(crate) fn on_event(&mut self, event: &Event) {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Right => self.next(),
                KeyCode::Left => self.previous(),
                _ => {}
            }
        }
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
        let plot = self.data.get(name).unwrap();

        Chart::<'a>::new(plot.datasets())
            .block(Block::default())
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .title("Iteration")
                    .labels(plot.labels_x.iter().map(|s| s.bold()).collect())
                    .bounds(plot.bounds_x),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .labels(plot.labels_y.iter().map(|s| s.bold()).collect())
                    .bounds(plot.bounds_y),
            )
    }
}

#[derive(new)]
pub(crate) enum NumericMetricView<'a> {
    Plots(&'a [String], usize, Chart<'a>),
    None,
}

impl<'a> NumericMetricView<'a> {
    pub(crate) fn render<'b>(self, frame: &mut TFrame<'b>, size: Rect) {
        match self {
            Self::Plots(titles, selected, chart) => {
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
                let plot_type = Paragraph::new(Line::from("Recent History".bold()))
                    .alignment(Alignment::Center);

                frame.render_widget(tabs, chunks[0]);
                frame.render_widget(plot_type, chunks[1]);
                frame.render_widget(chart, chunks[2]);
            }
            Self::None => {}
        };
    }
}
