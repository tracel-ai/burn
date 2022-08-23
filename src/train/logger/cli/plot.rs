use crate::train::metric::{Metric, MetricState, MetricStateDyn, NumericMetric};
use textplots::{Chart, Plot, Shape};

pub struct TextPlot<M: NumericMetric> {
    metric: M,
    values: Vec<f32>,
}

#[derive(new)]
pub struct TextPlotState {
    inner: MetricStateDyn,
    plot: String,
}

impl MetricState for TextPlotState {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn pretty(&self) -> String {
        format!("{}{}", self.inner.pretty(), self.plot)
    }

    fn serialize(&self) -> String {
        self.inner.serialize()
    }
}

impl<M: NumericMetric> TextPlot<M> {
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            values: Vec::new(),
        }
    }
}

impl<M, T> Metric<T> for TextPlot<M>
where
    M: Metric<T> + NumericMetric,
{
    fn update(&mut self, item: &T) -> MetricStateDyn {
        let state = self.metric.update(item);
        self.values.push(self.metric.value() as f32);

        let graph = Chart::new(256, 32, 0.0, self.values.len() as f32)
            .lineplot(&Shape::Lines(&smooth_values(&self.values, 256)))
            .to_string();

        Box::new(TextPlotState::new(state, format!("\n\n{}", graph)))
    }

    fn clear(&mut self) {
        self.metric.clear();
    }
}

fn smooth_values(values: &Vec<f32>, size_appox: usize) -> Vec<(f32, f32)> {
    let batch_size = values.len() / size_appox;
    if batch_size == 0 {
        return values
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f32, *v as f32))
            .collect();
    }

    let mut output = Vec::with_capacity(size_appox);
    let mut current_sum = 0.0;
    let mut current_count = 0;

    for value in values.iter() {
        current_sum += value;
        current_count += 1;

        if current_count >= batch_size {
            output.push(current_sum / current_count as f32);
        }
    }

    if current_count > 0 {
        output.push(current_sum / current_count as f32);
    }

    output
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f32, *v as f32))
        .collect()
}
