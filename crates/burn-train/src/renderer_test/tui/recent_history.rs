use super::PlotAxes;
use ratatui::{
    style::{Color, Style, Stylize},
    symbols,
    widgets::{Dataset, GraphType},
};

const FACTOR_BEFORE_RESIZE: usize = 2;

/// A plot that shows the recent history at full resolution.
pub(crate) struct RecentHistoryPlot {
    pub(crate) axes: PlotAxes,
    train: RecentHistoryPoints,
    valid: RecentHistoryPoints,
    max_samples: usize,
}

struct RecentHistoryPoints {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    cursor: usize,
    points: Vec<(f64, f64)>,
    max_samples: usize,
    factor_before_resize: usize,
}

impl RecentHistoryPlot {
    pub(crate) fn new(max_samples: usize) -> Self {
        Self {
            axes: PlotAxes::default(),
            train: RecentHistoryPoints::new(max_samples),
            valid: RecentHistoryPoints::new(max_samples),
            max_samples,
        }
    }

    pub(crate) fn push_train(&mut self, data: f64) {
        let (x_min, x_current) = self.x();

        self.train.push((x_current, data));
        self.train.update_cursor(x_min);
        self.valid.update_cursor(x_min);

        self.update_bounds();
    }

    pub(crate) fn push_valid(&mut self, data: f64) {
        let (x_min, x_current) = self.x();

        self.valid.push((x_current, data));
        self.valid.update_cursor(x_min);
        self.train.update_cursor(x_min);

        self.update_bounds();
    }

    pub(crate) fn datasets(&self) -> Vec<Dataset<'_>> {
        let mut datasets = Vec::with_capacity(2);

        if self.train.num_visible_points() > 0 {
            datasets.push(self.train.dataset("Train", Color::LightRed));
        }

        if self.valid.num_visible_points() > 0 {
            datasets.push(self.valid.dataset("Valid", Color::LightBlue));
        }

        datasets
    }

    fn x(&mut self) -> (f64, f64) {
        let x_current = f64::max(self.train.max_x, self.valid.max_x) + 1.0;
        let mut x_min = f64::min(self.train.min_x, self.valid.min_x);
        if x_current - x_min >= self.max_samples as f64 {
            x_min += 1.0;
        }

        (x_min, x_current)
    }

    fn update_bounds(&mut self) {
        self.axes.update_bounds(
            (self.train.min_x, self.train.max_x),
            (self.valid.min_x, self.valid.max_x),
            (self.train.min_y, self.train.max_y),
            (self.valid.min_y, self.valid.max_y),
        );
    }
}

impl RecentHistoryPoints {
    fn new(max_samples: usize) -> Self {
        let factor_before_resize = FACTOR_BEFORE_RESIZE;

        Self {
            min_x: 0.,
            max_x: 0.,
            min_y: f64::MAX,
            max_y: f64::MIN,
            points: Vec::with_capacity(factor_before_resize * max_samples),
            cursor: 0,
            max_samples,
            factor_before_resize,
        }
    }

    fn num_visible_points(&self) -> usize {
        self.points.len()
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
    }

    fn update_cursor(&mut self, min_x: f64) {
        if self.min_x >= min_x {
            return;
        }
        self.min_x = min_x;

        let mut update_y_max = false;
        let mut update_y_min = false;

        while let Some((x, y)) = self.points.get(self.cursor) {
            if *x >= self.min_x {
                break;
            }

            if *y == self.max_y {
                update_y_max = true
            }
            if *y == self.min_y {
                update_y_min = true;
            }

            self.cursor += 1;
        }

        if update_y_max {
            self.max_y = self.calculate_max_y();
        }

        if update_y_min {
            self.min_y = self.calculate_min_y();
        }

        if self.points.len() >= self.max_samples * self.factor_before_resize {
            self.resize();
        }
    }

    fn slice(&self) -> &[(f64, f64)] {
        &self.points[self.cursor..self.points.len()]
    }

    fn calculate_max_y(&self) -> f64 {
        let mut max_y = f64::MIN;

        for (_x, y) in self.slice() {
            if *y > max_y {
                max_y = *y;
            }
        }

        max_y
    }

    fn calculate_min_y(&self) -> f64 {
        let mut min_y = f64::MAX;

        for (_x, y) in self.slice() {
            if *y < min_y {
                min_y = *y;
            }
        }

        min_y
    }

    fn resize(&mut self) {
        let mut points = Vec::with_capacity(self.max_samples * self.factor_before_resize);

        for i in self.cursor..self.points.len() {
            points.push(self.points[i]);
        }

        self.points = points;
        self.cursor = 0;
    }

    fn dataset<'a>(&'a self, name: &'a str, color: Color) -> Dataset<'a> {
        let data = &self.points[self.cursor..self.points.len()];

        Dataset::default()
            .name(name)
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(color).bold())
            .graph_type(GraphType::Scatter)
            .data(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_update_bounds_max_y() {
        let mut chart = RecentHistoryPlot::new(3);
        chart.push_train(15.0);
        chart.push_train(10.0);
        chart.push_train(14.0);

        assert_eq!(chart.axes.bounds_y[1], 15.);
        chart.push_train(10.0);
        assert_eq!(chart.axes.bounds_y[1], 14.);
    }

    #[test]
    fn test_push_update_bounds_min_y() {
        let mut chart = RecentHistoryPlot::new(3);
        chart.push_train(5.0);
        chart.push_train(10.0);
        chart.push_train(14.0);

        assert_eq!(chart.axes.bounds_y[0], 5.);
        chart.push_train(10.0);
        assert_eq!(chart.axes.bounds_y[0], 10.);
    }
}
