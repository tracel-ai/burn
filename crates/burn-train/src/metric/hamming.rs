use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::{activation::sigmoid, backend::Backend, ElementConversion, Int, Tensor};

/// The hamming score, sometimes referred to as multi-label or label-based accuracy.
pub struct HammingScore<B: Backend> {
    state: NumericMetricState,
    threshold: f32,
    sigmoid: bool,
    _b: PhantomData<B>,
}

/// The [hamming score](HammingScore) input type.
#[derive(new)]
pub struct HammingScoreInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 2, Int>,
}

impl<B: Backend> HammingScore<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the sigmoid activation function usage.
    pub fn with_sigmoid(mut self, sigmoid: bool) -> Self {
        self.sigmoid = sigmoid;
        self
    }
}

impl<B: Backend> Default for HammingScore<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            threshold: 0.5,
            sigmoid: false,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for HammingScore<B> {
    type Input = HammingScoreInput<B>;

    fn update(&mut self, input: &HammingScoreInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, _n_classes] = input.outputs.dims();

        let targets = input.targets.clone();

        let mut outputs = input.outputs.clone();

        if self.sigmoid {
            outputs = sigmoid(outputs);
        }

        let score = outputs
            .greater_elem(self.threshold)
            .equal(targets.bool())
            .float()
            .mean()
            .into_scalar()
            .elem::<f64>();

        self.state.update(
            100.0 * score,
            batch_size,
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> String {
        format!("Hamming Score @ Threshold({})", self.threshold)
    }
}

impl<B: Backend> Numeric for HammingScore<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_hamming_score() {
        let device = Default::default();
        let mut metric = HammingScore::<TestBackend>::new();

        let x = Tensor::from_data(
            [
                [0.32, 0.52, 0.38, 0.68, 0.61], // with x > 0.5: [0, 1, 0, 1, 1]
                [0.43, 0.31, 0.21, 0.63, 0.53], //               [0, 0, 0, 1, 1]
                [0.44, 0.25, 0.71, 0.39, 0.73], //               [0, 0, 1, 0, 1]
                [0.49, 0.37, 0.68, 0.39, 0.31], //               [0, 0, 1, 0, 0]
            ],
            &device,
        );
        let y = Tensor::from_data(
            [
                [0, 1, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 0],
            ],
            &device,
        );

        let _entry = metric.update(
            &HammingScoreInput::new(x.clone(), y.clone()),
            &MetricMetadata::fake(),
        );
        assert_eq!(100.0, metric.value());

        // Invert all targets: y = (1 - y)
        let y = y.neg().add_scalar(1);
        let _entry = metric.update(
            &HammingScoreInput::new(x.clone(), y), // invert targets (1 - y)
            &MetricMetadata::fake(),
        );
        assert_eq!(0.0, metric.value());

        // Invert 5 target values -> 1 - (5/20) = 0.75
        let y = Tensor::from_data(
            [
                [0, 1, 1, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 1, 1, 0, 0],
            ],
            &device,
        );
        let _entry = metric.update(
            &HammingScoreInput::new(x, y), // invert targets (1 - y)
            &MetricMetadata::fake(),
        );
        assert_eq!(75.0, metric.value());
    }

    #[test]
    fn test_parameterized_unique_name() {
        let metric_a = HammingScore::<TestBackend>::new().with_threshold(0.5);
        let metric_b = HammingScore::<TestBackend>::new().with_threshold(0.75);
        let metric_c = HammingScore::<TestBackend>::new().with_threshold(0.5);

        assert_ne!(metric_a.name(), metric_b.name());
        assert_eq!(metric_a.name(), metric_c.name());
    }
}
