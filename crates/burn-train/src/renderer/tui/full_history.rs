use super::PlotAxes;
use crate::{
    metric::NumericEntry,
    renderer::tui::{TuiSplit, TuiTag},
};
use ratatui::{
    style::{Color, Style},
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
    /// Index of the first point of each line segment. A later point whose x is
    /// further than `step_size` from the previous one starts a new segment, so
    /// validation runs are not joined across the training steps between them.
    segment_starts: Vec<usize>,
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
    pub(crate) fn push(&mut self, tag: TuiTag, data: Option<NumericEntry>) {
        let x_current = self.next_x_state as f64;
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

        match data {
            Some(y) => {
                // If we already have batch points registered, ignore this for the line graph
                // to prevent an N+1 visual step artifact.
                if let NumericEntry::Final(val) = y
                    && !points.points.is_empty()
                {
                    points.set_final(val);
                    return;
                }

                self.next_x_state += 1;
                points.push((x_current, y));
            }
            None => {
                // Null step -> advance X to keep step alignment across metrics
                self.next_x_state += 1;
            }
        }

        self.update_bounds();
    }

    pub(crate) fn datasets(&self) -> Vec<Dataset<'_>> {
        let mut datasets = Vec::new();

        for (tag, points) in self.points.iter() {
            let color = tag.split.color();
            let name = format!("{tag}");
            let segments = points.segments();
            if segments.is_empty() {
                datasets.push(points.line_dataset(true, name, color, &points.points));
                continue;
            }
            for (index, segment) in segments.into_iter().enumerate() {
                datasets.push(points.line_dataset(index == 0, name.clone(), color, segment));
            }
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
            segment_starts: Vec::new(),
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
                aggregated_value,
                count,
            } => {
                self.avg_sum += aggregated_value * count as f64;
                self.avg_counter += count as f64;
                aggregated_value
            }
            NumericEntry::Final(val) => {
                // For global-only metrics (e.g., AUC-PR), this is our single valid line anchor
                self.set_final(val);
                val
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

        if let Some(&(last_x, _)) = self.points.last() {
            if x - last_x > self.step_size as f64 {
                self.segment_starts.push(self.points.len());
            }
        } else {
            self.segment_starts.push(0);
        }

        self.points.push((x, y));

        if self.points.len() > self.max_samples {
            self.resize();
        }
    }

    fn set_final(&mut self, val: f64) {
        // Final value does not affect plot points, but should affect the bar graph
        self.avg_sum = val;
        self.avg_counter = 1.0;
    }

    /// We keep only half the points and we double the step size.
    ///
    /// This ensure that we have the same amount of points across the X axis.
    fn resize(&mut self) {
        let old_starts = self.segment_starts.clone();
        let segment_index = |point_index: usize| {
            old_starts
                .partition_point(|&start| start <= point_index)
                .saturating_sub(1)
        };
        let mut points = Vec::with_capacity(self.max_samples / 2);
        let mut segment_starts = Vec::new();
        let mut current_segment = None;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;

        for (i, (x, y)) in self.points.drain(0..self.points.len()).enumerate() {
            if i % 2 == 0 {
                let segment = segment_index(i);
                if current_segment != Some(segment) {
                    segment_starts.push(points.len());
                    current_segment = Some(segment);
                }
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
        self.segment_starts = segment_starts;
        self.step_size *= 2;

        self.min_x = min_x;
        self.max_x = max_x;
        self.min_y = min_y;
        self.max_y = max_y;
    }

    fn segments(&self) -> Vec<&[(f64, f64)]> {
        if self.points.is_empty() {
            return Vec::new();
        }
        let starts = if self.segment_starts.is_empty() {
            vec![0]
        } else {
            self.segment_starts.clone()
        };
        let mut out = Vec::with_capacity(starts.len());
        for (idx, &start) in starts.iter().enumerate() {
            let end = starts.get(idx + 1).copied().unwrap_or(self.points.len());
            if start < end && start < self.points.len() {
                out.push(&self.points[start..end.min(self.points.len())]);
            }
        }
        out
    }

    fn line_dataset<'a>(
        &'a self,
        named: bool,
        name: String,
        color: Color,
        data: &'a [(f64, f64)],
    ) -> Dataset<'a> {
        let mut dataset = Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(color).bold())
            .graph_type(GraphType::Line)
            .data(data);
        if named {
            dataset = dataset.name(name);
        }
        dataset
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
                .label(label),
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
            chart.push(tag_train.clone(), Some(NumericEntry::Value(i as f64)));
        }
        for i in 0..60 {
            chart.push(tag_valid.clone(), Some(NumericEntry::Value(i as f64)));
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

    #[test]
    fn validation_runs_are_separate_line_segments() {
        let mut chart = FullHistoryPlot::new(100);
        let train = TuiTag::new(TuiSplit::Train, TuiGroup::Default);
        let valid = TuiTag::new(TuiSplit::Valid, TuiGroup::Default);

        chart.push(valid.clone(), Some(NumericEntry::Value(1.0)));
        chart.push(valid.clone(), Some(NumericEntry::Value(0.8)));
        chart.push(train.clone(), Some(NumericEntry::Value(2.0)));
        chart.push(train, Some(NumericEntry::Value(1.5)));
        chart.push(valid.clone(), Some(NumericEntry::Value(0.4)));

        let segments = chart.points.get(&valid).unwrap().segments();
        assert_eq!(
            segments.len(),
            2,
            "a gap in x must break the validation line"
        );
        assert_eq!(segments[0], &[(0.0, 1.0), (1.0, 0.8)]);
        assert_eq!(segments[1], &[(4.0, 0.4)]);
        // Train is one segment; validation is two. Only the first of each series is named.
        assert_eq!(chart.datasets().len(), 3);
    }

    #[test]
    fn test_points_with_none_gaps() {
        let mut chart = FullHistoryPlot::new(10);
        let tag_train = TuiTag::new(TuiSplit::Train, TuiGroup::Default);

        // Push 3 gap (None) steps -> should advance X ticks without adding points
        chart.push(tag_train.clone(), None);
        chart.push(tag_train.clone(), None);
        chart.push(tag_train.clone(), None);

        // Push valid point -> should land at X = 3.0
        chart.push(tag_train.clone(), Some(NumericEntry::Value(10.0)));

        let points = &chart.points.get(&tag_train).unwrap().points;

        // The line plot should only hold 1 actual coordinate
        assert_eq!(points.len(), 1);
        assert_eq!(points[0], (3.0, 10.0));
        assert_eq!(chart.next_x_state, 4);
    }

    #[test]
    fn test_points_final_ignored_when_points_exist() {
        let mut chart = FullHistoryPlot::new(10);
        let tag_train = TuiTag::new(TuiSplit::Train, TuiGroup::Default);

        // Push two normal batch points
        chart.push(tag_train.clone(), Some(NumericEntry::Value(1.0))); // X = 0.0
        chart.push(tag_train.clone(), Some(NumericEntry::Value(2.0))); // X = 1.0

        let expected_points = vec![(0.0, 1.0), (1.0, 2.0)];
        assert_eq!(chart.next_x_state, 2);
        assert_eq!(
            chart.points.get(&tag_train).unwrap().points,
            expected_points
        );

        // Push Final entry -> should update `final_value` but NOT advance next_x_state or add to points
        chart.push(tag_train.clone(), Some(NumericEntry::Final(1.5)));
        assert_eq!(chart.next_x_state, 2); // X state remains 2
        let history = chart.points.get(&tag_train).unwrap();
        assert_eq!(history.points.len(), 2);
        assert_eq!(history.points, expected_points);
        // The avg sum and counters are overwritten (used in bar graph)
        assert_eq!(history.avg_sum, 1.5);
        assert_eq!(history.avg_counter, 1.0);
    }
}
