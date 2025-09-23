use super::PlotAxes;
use crate::{
    metric::NumericEntry,
    renderer::tui::{TuiSplit, TuiTag},
};
use ratatui::{
    style::{Color, Style, Stylize},
    symbols,
    widgets::{Bar, Dataset, GraphType},
};
use std::collections::BTreeMap;

/// A plot that shows the full history at a reduced resolution.
pub(crate) struct FullHistoryPlot {
    pub(crate) axes: PlotAxes,
    points: BTreeMap<TuiTag, FullHistoryPoints>,
    max_samples: usize,
    max_samples_ratio: BTreeMap<TuiSplit, f64>,
    next_x_state: usize,
}

struct FullHistoryPoints {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    avg_sum: f64,
    avg_counter: f64,
    points: Vec<(f64, f64)>,
    max_samples: usize,
    step_size: usize,
}

impl FullHistoryPlot {
    /// Create a new history plot.
    pub(crate) fn new(max_samples: usize) -> Self {
        Self {
            points: BTreeMap::default(),
            axes: PlotAxes::default(),
            max_samples,
            max_samples_ratio: BTreeMap::default(),
            next_x_state: 0,
        }
    }

    /// Update the maximum amount of sample to display for the validation points.
    ///
    /// This is necessary if we want the validation line to have the same point density as the
    /// training line.
    pub(crate) fn update_max_sample(&mut self, split: TuiSplit, ratio: f64) {
        self.max_samples_ratio.insert(split, ratio);

        self.points
            .iter_mut()
            .filter(|(tag, _)| tag.split == split)
            .for_each(|(_, points)| {
                points.max_samples = (self.max_samples as f64 * ratio) as usize;
            });
    }

    /// Register a training data point.
    pub(crate) fn push(&mut self, tag: TuiTag, data: NumericEntry) {
        let x_current = self.next_x();
        let points = match self.points.get_mut(&tag) {
            Some(val) => val,
            None => {
                let max_samples = self
                    .max_samples_ratio
                    .get(&tag.split)
                    .map(|ratio| (*ratio * self.max_samples as f64) as usize)
                    .unwrap_or(self.max_samples);
                self.points
                    .insert(tag.clone(), FullHistoryPoints::new(max_samples));
                self.points.get_mut(&tag).unwrap()
            }
        };

        points.push((x_current, data));

        self.update_bounds();
    }

    pub(crate) fn datasets(&self) -> Vec<Dataset<'_>> {
        let mut datasets = Vec::with_capacity(2);

        for (tag, points) in self.points.iter() {
            datasets.push(points.dataset(format!("{tag}"), tag.split.color()));
        }

        datasets
    }

    pub(crate) fn bars(&self, max: u64, bar_width: &mut usize) -> Vec<Bar<'_>> {
        let mut bars = Vec::new();

        for (tag, points) in self.points.iter() {
            if let Some((bar, width)) = points.bar(tag, max) {
                *bar_width = usize::max(*bar_width, width);
                bars.push(bar);
            }
        }

        bars
    }

    fn next_x(&mut self) -> f64 {
        let value = self.next_x_state;
        self.next_x_state += 1;
        value as f64
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

impl FullHistoryPoints {
    fn new(max_samples: usize) -> Self {
        Self {
            min_x: 0.,
            max_x: 0.,
            min_y: f64::MAX,
            max_y: f64::MIN,
            avg_sum: 0.0,
            avg_counter: 0.0,
            points: Vec::with_capacity(max_samples),
            max_samples,
            step_size: 1,
        }
    }

    fn push(&mut self, (x, y): (f64, NumericEntry)) {
        if !(x as usize).is_multiple_of(self.step_size) {
            return;
        }

        let y = match y {
            NumericEntry::Value(val) => {
                self.avg_sum += val;
                self.avg_counter += 1.0;
                val
            }
            NumericEntry::Aggregated {
                sum,
                count,
                current,
            } => {
                self.avg_sum = sum;
                self.avg_counter = count as f64;
                current
            }
        };

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

    /// We keep only half the points and we double the step size.
    ///
    /// This ensure that we have the same amount of points across the X axis.
    fn resize(&mut self) {
        let mut points = Vec::with_capacity(self.max_samples / 2);
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;

        for (i, (x, y)) in self.points.drain(0..self.points.len()).enumerate() {
            if i % 2 == 0 {
                if x > max_x {
                    max_x = x;
                }
                if x < min_x {
                    min_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
                if y < min_y {
                    min_y = y;
                }

                points.push((x, y));
            }
        }

        self.points = points;
        self.step_size *= 2;

        self.min_x = min_x;
        self.max_x = max_x;
        self.min_y = min_y;
        self.max_y = max_y;
    }

    fn dataset<'a>(&'a self, name: String, color: Color) -> Dataset<'a> {
        Dataset::default()
            .name(name)
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(color).bold())
            .graph_type(GraphType::Line)
            .data(&self.points)
    }

    fn bar<'a>(&'a self, tag: &TuiTag, max: u64) -> Option<(Bar<'a>, usize)> {
        if self.avg_sum == 0.0 {
            return None;
        }

        let label = format!("{tag}");
        let width = usize::max(label.len(), 7); // 7 min width

        let factor = max as f64;

        let avg = self.avg_sum / self.avg_counter;

        Some((
            Bar::default()
                .value((avg * factor) as u64)
                .style(tag.split.color())
                .text_value(format!("{:.2}", avg))
                .label(label.into()),
            width,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::tui::{TuiGroup, TuiSplit};

    #[test]
    fn test_points() {
        let mut chart = FullHistoryPlot::new(10);
        let tag_train = TuiTag::new(TuiSplit::Train, TuiGroup::Default);
        let tag_valid = TuiTag::new(TuiSplit::Valid, TuiGroup::Default);
        chart.update_max_sample(tag_valid.split, 0.6);

        for i in 0..100 {
            chart.push(tag_train.clone(), NumericEntry::Value(i as f64));
        }
        for i in 0..60 {
            chart.push(tag_valid.clone(), NumericEntry::Value(i as f64));
        }

        let expected_train = vec![
            (0.0, 0.0),
            (16.0, 16.0),
            (32.0, 32.0),
            (48.0, 48.0),
            (64.0, 64.0),
            (80.0, 80.0),
            (96.0, 96.0),
        ];

        let expected_valid = vec![(100.0, 0.0), (116.0, 16.0), (128.0, 28.0), (144.0, 44.0)];

        assert_eq!(
            chart.points.get(&tag_train).unwrap().points,
            expected_train,
            "Expected train data points"
        );
        assert_eq!(
            chart.points.get(&tag_valid).unwrap().points,
            expected_valid,
            "Expected valid data points"
        );
    }
}
