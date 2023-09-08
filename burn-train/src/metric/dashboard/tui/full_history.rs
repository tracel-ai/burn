use ratatui::{
    style::{Color, Style, Stylize},
    symbols,
    widgets::{Dataset, GraphType},
};

use crate::metric::format_float;

static FACTOR_BEFORE_RESIZE: usize = 2;
static AXIS_TITLE_PRECISION: usize = 2;

pub(crate) struct FullHistoryPlot {
    pub(crate) labels_x: Vec<String>,
    pub(crate) labels_y: Vec<String>,
    pub(crate) bounds_x: [f64; 2],
    pub(crate) bounds_y: [f64; 2],
    train: FullHistoryPoints,
    valid: FullHistoryPoints,
}

struct FullHistoryPoints {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    points: Vec<(f64, f64)>,
    max_samples: usize,
    factor_resize: usize,
}

impl FullHistoryPlot {
    pub(crate) fn new(max_samples: usize) -> Self {
        Self {
            bounds_x: [f64::MAX, f64::MIN],
            bounds_y: [f64::MAX, f64::MIN],
            labels_x: Vec::new(),
            labels_y: Vec::new(),
            train: FullHistoryPoints::new(max_samples),
            valid: FullHistoryPoints::new(max_samples),
        }
    }

    pub(crate) fn push_train(&mut self, data: f64) {
        let x_current = self.x();
        self.train.push((x_current, data));

        self.update_bounds();
    }

    pub(crate) fn push_valid(&mut self, data: f64) {
        let x_current = self.x();

        self.valid.push((x_current, data));

        self.update_bounds();
    }

    pub(crate) fn datasets<'a>(&'a self) -> Vec<Dataset<'a>> {
        let mut datasets = Vec::with_capacity(2);

        if !self.train.is_empty() {
            datasets.push(self.train.dataset("Train", Color::LightRed));
        }

        if !self.valid.is_empty() {
            datasets.push(self.valid.dataset("Valid", Color::LightBlue));
        }

        datasets
    }
    fn x(&mut self) -> f64 {
        f64::max(self.train.max_x, self.valid.max_x) + 1.0
    }

    fn update_bounds(&mut self) {
        let x_min = f64::min(self.train.min_x, self.valid.min_x);
        let x_max = f64::max(self.train.max_x, self.valid.max_x);
        let y_min = f64::min(self.train.min_y, self.valid.min_y);
        let y_max = f64::max(self.train.max_y, self.valid.max_y);

        self.bounds_x = [x_min, x_max];
        self.bounds_y = [y_min, y_max];

        // We know x are integers.
        self.labels_x = vec![format!("{x_min}"), format!("{x_max}")];
        self.labels_y = vec![
            format_float(y_min, AXIS_TITLE_PRECISION),
            format_float(y_max, AXIS_TITLE_PRECISION),
        ];
    }
}

impl FullHistoryPoints {
    fn new(max_samples: usize) -> Self {
        Self {
            min_x: 0.,
            max_x: 0.,
            min_y: f64::MAX,
            max_y: f64::MIN,
            points: Vec::with_capacity(max_samples),
            max_samples,
            factor_resize: FACTOR_BEFORE_RESIZE,
        }
    }

    fn push(&mut self, (x, y): (f64, f64)) {
        if x > self.max_x {
            self.max_x = x;
        }
        if x < self.min_x {
            self.min_x = x;
        }
        if y > self.max_y {
            self.max_y = y;
        }
        if y < self.min_y {
            self.min_y = y
        }
        self.points.push((x, y));

        if self.points.len() > self.max_samples {
            self.resize();
        }
    }

    fn resize(&mut self) {
        let mut points = Vec::with_capacity(self.max_samples / self.factor_resize);
        let mut current_x = 0.;
        let mut current_y = 0.;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;

        for (i, (x, y)) in self.points.drain(0..self.points.len()).enumerate() {
            current_x += x;
            current_y += y;

            if (i + 1) % 2 == 0 {
                let new_x = current_x / 2.;
                let new_y = current_y / 2.;
                if new_x > max_x {
                    max_x = new_x;
                }
                if new_x < min_x {
                    min_x = new_x;
                }
                if new_y > max_y {
                    max_y = new_y;
                }
                if new_y < min_y {
                    min_y = new_y;
                }
                points.push((new_x, new_y));
                current_x = 0.;
                current_y = 0.;
            }
        }

        self.points = points;
        self.min_x = min_x;
        self.max_x = max_x;
        self.min_y = min_y;
        self.max_y = max_y;
    }

    fn dataset<'a>(&'a self, name: &'a str, color: Color) -> Dataset<'a> {
        Dataset::default()
            .name(name)
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(color).bold())
            .graph_type(GraphType::Line)
            .data(&self.points)
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_update_bounds_max_y() {
        let mut chart = FullHistoryPlot::new(3);
        chart.push_train(15.0);
        chart.push_train(10.0);
        chart.push_train(14.0);

        assert_eq!(chart.bounds_y[1], 15.);
        chart.push_train(10.0);
        assert_eq!(chart.bounds_y[1], 14.);
    }

    #[test]
    fn test_push_update_bounds_min_y() {
        let mut chart = FullHistoryPlot::new(3);
        chart.push_train(5.0);
        chart.push_train(10.0);
        chart.push_train(14.0);

        assert_eq!(chart.bounds_y[0], 5.);
        chart.push_train(10.0);
        assert_eq!(chart.bounds_y[0], 10.);
    }
}
