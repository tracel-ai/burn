use core::marker::PhantomData;
use std::sync::Arc;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricMetadata, SerializedEntry};
use crate::metric::{
    Metric, MetricAttributes, MetricName, Numeric, NumericAttributes, NumericEntry,
};
use burn_core::tensor::{ElementConversion, Int, Tensor, activation::sigmoid, backend::Backend};

/// The hamming score, sometimes referred to as multi-label or label-based accuracy.
#[derive(Clone)]
pub struct HammingScore<B: Backend> {
    name: MetricName,
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

    fn update_name(&mut self) {
        self.name = Arc::new(format!("Hamming Score @ Threshold({})", self.threshold));
    }

    /// Sets the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self.update_name();
        self
    }

    /// Sets the sigmoid activation function usage.
    pub fn with_sigmoid(mut self, sigmoid: bool) -> Self {
        self.sigmoid = sigmoid;
        self.update_name();
        self
    }
}

impl<B: Backend> Default for HammingScore<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        let threshold = 0.5;
        let name = Arc::new(format!("Hamming Score @ Threshold({})", threshold));

        Self {
            name,
            state: NumericMetricState::default(),
            threshold,
            sigmoid: false,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for HammingScore<B> {
    type Input = HammingScoreInput<B>;

    fn update(
        &mut self,
        input: &HammingScoreInput<B>,
        _metadata: &MetricMetadata,
    ) -> SerializedEntry {
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

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Numeric for HammingScore<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
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
        assert_eq!(100.0, metric.value().current());

        // Invert all targets: y = (1 - y)
        let y = y.neg().add_scalar(1);
        let _entry = metric.update(
            &HammingScoreInput::new(x.clone(), y), // invert targets (1 - y)
            &MetricMetadata::fake(),
        );
        assert_eq!(0.0, metric.value().current());

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
        assert_eq!(75.0, metric.value().current());
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
