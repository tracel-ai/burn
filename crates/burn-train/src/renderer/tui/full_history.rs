use std::collections::HashMap;

use crate::renderer::tui::{TuiGroup, TuiSplit};

use super::PlotAxes;
use ratatui::{
    style::{Color, Style, Stylize},
    symbols,
    widgets::{Bar, Dataset, GraphType},
};

/// A plot that shows the full history at a reduced resolution.
pub(crate) struct FullHistoryPlot {
    pub(crate) axes: PlotAxes,
    points: HashMap<(TuiSplit, TuiGroup), FullHistoryPoints>,
    max_samples: usize,
    next_x_state: usize,
}

struct FullHistoryPoints {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    avg: f64,
    avg_counter: f64,
    points: Vec<(f64, f64)>,
    max_samples: usize,
    step_size: usize,
}

impl FullHistoryPlot {
    /// Create a new history plot.
    pub(crate) fn new(max_samples: usize) -> Self {
        Self {
            points: HashMap::default(),
            axes: PlotAxes::default(),
            max_samples,
            next_x_state: 0,
        }
    }

    /// Update the maximum amount of sample to display for the validation points.
    ///
    /// This is necessary if we want the validation line to have the same point density as the
    /// training line.
    pub(crate) fn update_max_sample_valid(&mut self, ratio_train: f64) {
        // if self.valid.step_size == 1 {
        //     self.valid.max_samples = (ratio_train * self.train.max_samples as f64) as usize;
        // }
    }

    /// Update the maximum amount of sample to display for the testing points.
    ///
    /// This is necessary if we want the testing line to have the same point density as the
    /// training line.
    pub(crate) fn update_max_sample_test(&mut self, ratio_train: f64) {
        // if self.test.step_size == 1 {
        //     self.test.max_samples = (ratio_train * self.train.max_samples as f64) as usize;
        // }
    }

    /// Register a training data point.
    pub(crate) fn push(&mut self, split: TuiSplit, group: TuiGroup, data: f64) {
        let key = (split, group);

        let x_current = self.next_x();
        let points = match self.points.get_mut(&key) {
            Some(val) => val,
            None => {
                self.points
                    .insert(key.clone(), FullHistoryPoints::new(self.max_samples));
                self.points.get_mut(&key).unwrap()
            }
        };

        points.push((x_current, data));

        self.update_bounds();
    }

    pub(crate) fn datasets(&self) -> Vec<Dataset<'_>> {
        let mut datasets = Vec::with_capacity(2);

        for ((split, group), points) in self.points.iter() {
            let color = match split {
                TuiSplit::Train => Color::LightRed,
                TuiSplit::Valid => Color::LightBlue,
                TuiSplit::Test => Color::LightGreen,
            };
            datasets.push(points.dataset(format!("{group}{split}"), color));
        }

        datasets
    }

    pub(crate) fn reset_avg(&mut self) {
        self.points.values_mut().for_each(|points| {
            points.avg = 0.0;
            points.avg_counter = 0.0;
        });
    }

    pub(crate) fn bars(&self, min: u64, max: u64) -> Vec<Bar<'_>> {
        let mut bars = Vec::new();

        for ((split, group), points) in self.points.iter() {
            if let Some(bar) = points.bar(split.color(), min, max) {
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
        self.axes.update_bounds(
            (self.train.min_x, self.train.max_x),
            (self.valid.min_x, self.valid.max_x),
            (self.test.min_x, self.test.max_x),
            (self.train.min_y, self.train.max_y),
            (self.valid.min_y, self.valid.max_y),
            (self.test.min_y, self.test.max_y),
        );
    }
}

impl FullHistoryPoints {
    fn new(max_samples: usize) -> Self {
        Self {
            min_x: 0.,
            max_x: 0.,
            min_y: f64::MAX,
            max_y: f64::MIN,
            avg: 0.0,
            avg_counter: 0.0,
            points: Vec::with_capacity(max_samples),
            max_samples,
            step_size: 1,
        }
    }

    fn push(&mut self, (x, y): (f64, f64)) {
        if x as usize % self.step_size != 0 {
            return;
        }

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

        self.avg += y;
        self.avg_counter += 1.0;

        // For numerical stability. Can accumulate for ever.
        if self.avg_counter >= 1000.0 {
            self.avg = self.avg / self.avg_counter;
            self.avg_counter = 1.0;
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

    fn bar<'a>(&'a self, color: Color, min: u64, max: u64) -> Option<Bar<'a>> {
        if self.avg == 0.0 {
            return None;
        }

        // let range_current = self.max_y - self.min_y;
        // let range_expected = max - min;
        let factor = max as f64;

        let avg = self.avg / self.avg_counter;

        Some(
            Bar::default()
                .value((avg * factor) as u64)
                .style(color)
                .text_value(format!("{:.2}", avg))
                .label("Avg".into()),
        )
    }

    fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_points() {
    //     let mut chart = FullHistoryPlot::new(10);
    //     chart.update_max_sample_valid(0.6);

    //     for i in 0..100 {
    //         chart.push_train(i as f64);
    //     }
    //     for i in 0..60 {
    //         chart.push_valid(i as f64);
    //     }

    //     let expected_train = vec![
    //         (0.0, 0.0),
    //         (16.0, 16.0),
    //         (32.0, 32.0),
    //         (48.0, 48.0),
    //         (64.0, 64.0),
    //         (80.0, 80.0),
    //         (96.0, 96.0),
    //     ];

    //     let expected_valid = vec![(100.0, 0.0), (116.0, 16.0), (128.0, 28.0), (144.0, 44.0)];

    //     assert_eq!(chart.train.points, expected_train);
    //     assert_eq!(chart.valid.points, expected_valid);
    // }
}
