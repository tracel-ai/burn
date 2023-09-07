use crossterm::event::{Event, KeyCode};
use ratatui::{
    prelude::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    symbols,
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Tabs},
};
use std::collections::HashMap;

use super::TFrame;

#[derive(Default)]
pub(crate) struct NumericMetricsState {
    data: HashMap<String, ChartData>,
    names: Vec<String>,
    selected: usize,
}

impl NumericMetricsState {
    pub(crate) fn push_train(&mut self, key: String, data: f64) {
        if let Some(existing) = self.data.get_mut(&key) {
            existing.push_train(data);
        } else {
            let mut chart = ChartData::default();
            chart.push_train(data);
            self.names.push(key.clone());
            self.data.insert(key, chart);
        }
    }

    pub(crate) fn push_valid(&mut self, key: String, data: f64) {
        if let Some(existing) = self.data.get_mut(&key) {
            existing.push_valid(data);
        } else {
            let mut chart = ChartData::default();
            chart.push_valid(data);
            self.data.insert(key, chart);
        }
    }

    pub(crate) fn view<'a>(&'a self) -> NumericMetricView<'a> {
        NumericMetricView::new(&self.names, self.selected, self.chart())
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
        let data = self.data.get(name).unwrap();

        Chart::<'a>::new(data.to_datasets())
            .block(Block::default())
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .title("Iteration")
                    .labels(data.labels_x.iter().map(|s| s.bold()).collect())
                    .bounds(data.bounds_x),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .labels(data.labels_y.iter().map(|s| s.bold()).collect())
                    .bounds(data.bounds_y),
            )
    }
}

#[derive(new)]
pub(crate) struct NumericMetricView<'a> {
    titles: &'a [String],
    selected: usize,
    chart: Chart<'a>,
}

impl<'a> NumericMetricView<'a> {
    pub(crate) fn render<'b>(self, frame: &mut TFrame<'b>, size: Rect) {
        let block = Block::default().borders(Borders::ALL).title("Plots");
        let size_new = block.inner(size);
        frame.render_widget(block, size);

        let size = size_new;

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
            .split(size);

        let titles = self
            .titles
            .iter()
            .map(|i| Line::from(vec![i.yellow()]))
            .collect();

        let tabs = Tabs::new(titles)
            .block(Block::default())
            .select(self.selected)
            .style(Style::default())
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .bg(Color::Black),
            );

        frame.render_widget(tabs, chunks[0]);
        frame.render_widget(self.chart, chunks[1]);
    }
}

struct ChartData {
    bounds_x: [f64; 2],
    bounds_y: [f64; 2],
    labels_x: Vec<String>,
    labels_y: Vec<String>,
    train: Vec<(f64, f64)>,
    valid: Vec<(f64, f64)>,
    max_samples: usize,
}

impl Default for ChartData {
    fn default() -> Self {
        Self {
            bounds_x: [f64::MAX, f64::MIN],
            bounds_y: [f64::MAX, f64::MIN],
            labels_x: Vec::new(),
            labels_y: Vec::new(),
            train: Vec::new(),
            valid: Vec::new(),
            max_samples: 1000,
        }
    }
}

impl ChartData {
    fn push_train(&mut self, data: f64) {
        let last = self.last_iteration();

        Self::push(
            last,
            data,
            &mut self.train,
            &mut self.bounds_x,
            &mut self.bounds_y,
            self.max_samples,
        );

        self.update_bounds((last + 1.0, data));
    }

    fn push_valid(&mut self, data: f64) {
        let last = self.last_iteration();

        Self::push(
            last,
            data,
            &mut self.valid,
            &mut self.bounds_x,
            &mut self.bounds_y,
            self.max_samples,
        );

        self.update_bounds((last + 1.0, data));
    }

    fn push(
        iteration: f64,
        data: f64,
        items: &mut Vec<(f64, f64)>,
        bounds_x: &mut [f64; 2],
        bounds_y: &mut [f64; 2],
        max_samples: usize,
    ) {
        let data = (iteration + 1.0, data);
        items.push(data);

        if items.len() > max_samples {
            let (x, y) = items.remove(0);
            bounds_x[0] = x; // We know we always remove the minimum.

            let y_is_min = y.floor() == bounds_y[0];
            let y_is_max = y.ceil() == bounds_y[1];

            if y_is_min {
                bounds_y[0] = items
                    .iter()
                    .map(|(_x, y)| y.floor() as i64)
                    .min()
                    .unwrap_or(i64::MIN) as f64;
            }

            if y_is_max {
                bounds_y[1] = items
                    .iter()
                    .map(|(_x, y)| y.ceil() as i64)
                    .max()
                    .unwrap_or(i64::MAX) as f64;
            }
        }
    }

    fn last_iteration(&self) -> f64 {
        let last_valid = self.valid.last().map(|(x, _y)| *x).unwrap_or(0.0);
        let last_train = self.train.last().map(|(x, _y)| *x).unwrap_or(0.0);

        f64::max(last_train, last_valid)
    }

    fn update_bounds(&mut self, data: (f64, f64)) {
        if data.0 < self.bounds_x[0] {
            self.bounds_x[0] = data.0.floor();
        }
        if data.0 > self.bounds_x[1] {
            self.bounds_x[1] = data.0.floor();
        }
        if data.1 < self.bounds_y[0] {
            self.bounds_y[0] = data.1.ceil();
        }
        if data.1 > self.bounds_y[1] {
            self.bounds_y[1] = data.1.ceil();
        }

        self.labels_x = vec![self.bounds_x[0].to_string(), self.bounds_x[1].to_string()];
        self.labels_y = vec![self.bounds_y[0].to_string(), self.bounds_y[1].to_string()];
    }

    fn to_datasets<'a>(&'a self) -> Vec<Dataset<'a>> {
        let mut datasets = Vec::new();

        if !self.train.is_empty() {
            datasets.push(
                Dataset::default()
                    .name("Train")
                    .marker(symbols::Marker::Dot)
                    .style(Style::default().fg(Color::LightRed).bold())
                    .graph_type(GraphType::Scatter)
                    .data(&self.train),
            );
        }

        if !self.valid.is_empty() {
            datasets.push(
                Dataset::default()
                    .name("Valid")
                    .marker(symbols::Marker::Dot)
                    .style(Style::default().fg(Color::LightBlue).bold())
                    .graph_type(GraphType::Scatter)
                    .data(&self.valid),
            );
        }

        datasets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_update_bounds_max_y() {
        let mut chart = ChartData::default();
        chart.max_samples = 3;
        chart.push_train(15.0);
        chart.push_train(10.0);
        chart.push_train(14.0);

        assert_eq!(chart.bounds_y[1], 15.);
        chart.push_train(10.0);
        assert_eq!(chart.bounds_y[1], 14.);
    }

    #[test]
    fn test_push_update_bounds_min_y() {
        let mut chart = ChartData::default();
        chart.max_samples = 3;
        chart.push_train(5.0);
        chart.push_train(10.0);
        chart.push_train(14.0);

        assert_eq!(chart.bounds_y[0], 5.);
        chart.push_train(10.0);
        assert_eq!(chart.bounds_y[0], 10.);
    }
}
