use super::PlotAxes;
use crate::{metric::NumericEntry, renderer::tui::TuiTag};
use ratatui::{
    style::{Color, Style},
    symbols,
    widgets::{Dataset, GraphType},
};
use std::collections::BTreeMap;

const FACTOR_BEFORE_RESIZE: usize = 2;

/// A plot that shows the recent history at full resolution.
pub(crate) struct RecentHistoryPlot {
    pub(crate) axes: PlotAxes,
    points: BTreeMap<TuiTag, RecentHistoryPoints>,
    max_samples: usize,
    next_x_state: usize,
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
            points: BTreeMap::default(),
            max_samples,
            next_x_state: 0,
        }
    }

    pub(crate) fn push(&mut self, tag: TuiTag, data: Option<NumericEntry>) {
        if !self.points.contains_key(&tag) {
            self.points
                .insert(tag.clone(), RecentHistoryPoints::new(self.max_samples));
        }

        // Convert the NumericEntry into a plottable f64.
        // We filter out `Final` if we already have batch points for this tag.
        let plot_value = match data {
            Some(NumericEntry::Final(val)) => {
                let has_points = !self.points.get(&tag).unwrap().points.is_empty();
                if has_points {
                    return; // Ignore the N+1 artifact
                } else {
                    Some(val) // Fallback for global-only metrics
                }
            }
            Some(entry) => Some(entry.current()),
            None => None,
        };

        let x_current = self.next_x_state as f64;
        self.next_x_state += 1;

        let x_min = if self.next_x_state > self.max_samples {
            (self.next_x_state - self.max_samples) as f64
        } else {
            0.0
        };

        for (s, entry) in self.points.iter_mut() {
            if let Some(y) = plot_value {
                if s == &tag {
                    entry.push((x_current, y));
                }
            }
            entry.update_cursor(x_min);
        }

        self.update_bounds();
    }

    pub(crate) fn datasets(&self) -> Vec<Dataset<'_>> {
        let mut datasets = Vec::new();

        for (tag, points) in self.points.iter() {
            datasets.push(points.dataset(format!("{tag}"), tag.split.color()));
        }

        datasets
    }

    fn update_bounds(&mut self) {
        let (mut x_min, mut x_max) = (f64::MAX, f64::MIN);
        let (mut y_min, mut y_max) = (f64::MAX, f64::MIN);

        for points in self.points.values() {
            x_min = f64::min(x_min, points.min_x);
            x_max = f64::max(x_max, points.max_x);
            y_min = f64::min(y_min, points.min_y);
            y_max = f64::max(y_max, points.max_y);
        }

        self.axes.update_bounds((x_min, x_max), (y_min, y_max));
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
            max_y = f64::max(max_y, *y);
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

    fn dataset<'a>(&'a self, name: String, color: Color) -> Dataset<'a> {
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
    use crate::renderer::tui::{TuiGroup, TuiSplit};

    use super::*;

    #[test]
    fn test_push_update_bounds_max_y() {
        let mut chart = RecentHistoryPlot::new(3);
        let tag = TuiTag::new(TuiSplit::Train, TuiGroup::Default);

        chart.push(tag.clone(), Some(NumericEntry::Value(15.0)));
        chart.push(tag.clone(), Some(NumericEntry::Value(10.0)));
        chart.push(tag.clone(), Some(NumericEntry::Value(14.0)));

        assert_eq!(chart.axes.bounds_y[1], 15.);
        chart.push(tag, Some(NumericEntry::Value(10.0)));
        assert_eq!(chart.axes.bounds_y[1], 14.);
    }

    #[test]
    fn test_push_update_bounds_min_y() {
        let mut chart = RecentHistoryPlot::new(3);
        let tag = TuiTag::new(TuiSplit::Train, TuiGroup::Default);

        chart.push(tag.clone(), Some(NumericEntry::Value(5.0)));
        chart.push(tag.clone(), Some(NumericEntry::Value(10.0)));
        chart.push(tag.clone(), Some(NumericEntry::Value(14.0)));

        assert_eq!(chart.axes.bounds_y[0], 5.);
        chart.push(tag, Some(NumericEntry::Value(10.0)));
        assert_eq!(chart.axes.bounds_y[0], 10.);
    }

    #[test]
    fn test_push_update_no_value_bounds_min_x() {
        let mut chart = RecentHistoryPlot::new(2);
        let tag = TuiTag::new(TuiSplit::Train, TuiGroup::Default);

        // Push 1: None -> tick 0. bounds_x = (0, 0)
        chart.push(tag.clone(), None);
        assert_eq!(chart.axes.bounds_x, [0.0, 0.0]);

        // Push 2: None -> tick 1. bounds_x unchanged because no point was actually added
        chart.push(tag.clone(), None);
        assert_eq!(chart.axes.bounds_x, [0.0, 0.0]);

        // Push 3: Some(5.0) -> tick 2. Sliding window max_samples=2 -> bounds_x = (1, 2)
        chart.push(tag.clone(), Some(NumericEntry::Value(5.0)));
        assert_eq!(chart.axes.bounds_x, [1.0, 2.0]);
        assert_eq!(chart.axes.bounds_y, [5.0, 5.0]);

        // Push 4: Some(10.0) -> tick 3. bounds_x = (2, 3)
        chart.push(tag, Some(NumericEntry::Value(10.0)));
        assert_eq!(chart.axes.bounds_x, [2.0, 3.0]);
        assert_eq!(chart.axes.bounds_y, [5.0, 10.0]);
    }
}
