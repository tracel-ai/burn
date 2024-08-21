use core::marker::PhantomData;
use std::ops::Deref;
use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric, confusion_matrix::ConfusionMatrix};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Bool, ElementConversion, Tensor};

#[derive(Clone)]
enum MetricAverage {
    Micro,
    Macro,
    Weighted(Box<[f64]>)
}

/// The precision metric.
pub struct PrecisionMetric<B: Backend> {
    state: NumericMetricState,
    threshold: f32,
    average: MetricAverage,
    _b: PhantomData<B>,
}

/// The [precision metric](PrecisionMetric) input type.
#[derive(new)]
pub struct PrecisionInput<B: Backend> {
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2, Bool>,
}

impl<B: Backend> PrecisionMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets average type.
    pub fn with_average(mut self, average: MetricAverage) -> Self {
        self.average = average;
        self
    }

}

impl<B: Backend> Default for PrecisionMetric<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            threshold: 0.5,
            average: MetricAverage::Micro,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for PrecisionMetric<B> {
    const NAME: &'static str = "Precision";

    type Input = PrecisionInput<B>;

    fn update(&mut self, input: &PrecisionInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, n_classes] = input.predictions.dims();

        let targets = input.targets.clone().to_device(&B::Device::default());
        let predictions = input
            .predictions
            .clone()
            .to_device(&B::Device::default())
            .greater_elem(self.threshold);

        let stats = ConfusionMatrix::from(predictions, targets);

        let precision: f64 = match self.average.clone() {
            MetricAverage::Macro => {
                (stats.clone().true_positive.int().sum_dim(0) / stats.predicted_positive().int().sum_dim(0)).sum().into_scalar().elem::<f64>() / n_classes as f64
            },
            MetricAverage::Micro => {
                stats.clone().true_positive.int().sum().into_scalar().elem::<f64>() / stats.predicted_positive().int().sum().into_scalar().elem::<f64>()
            },
            MetricAverage::Weighted(weights) => {
                (stats.clone().true_positive.float().sum_dim(0) * Tensor::from_floats(weights.deref(), &B::Device::default())).sum().into_scalar().elem::<f64>()
            }
        };

        self.state.update(
            100.0 * precision,
            batch_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for PrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_precision_without_padding() {
        let device = Default::default();
        let mut metric = PrecisionMetric::<TestBackend>::new();
        let input = PrecisionInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.2, 0.8], // 2
                    [1.0, 0.0, 0.0], // 0
                    [0.3, 0.6, 0.2], // 1
                    [0.1, 0.7, 0.2], // 1
                ],
                &device,
            ),
            Tensor::from_data(
                [
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ],
                &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(75.0, metric.value());
    }

    /*#[test]
    fn test_precision_with_padding() {
        let device = Default::default();
        let mut metric = PrecisionMetric::<TestBackend>::new().with_pad_token(3);
        let input = PrecisionInput::n(
            Tensor::from_data(
                [
                    [0.0, 0.2, 0.8, 0.0], // 2
                    [1.0, 2.0, 0.5, 0.0], // 1
                    [0.4, 0.1, 0.2, 0.0], // 0
                    [0.6, 0.7, 0.2, 0.0], // 1
                    [0.0, 0.1, 0.2, 5.0], // Predicted padding should not count
                    [0.0, 0.1, 0.2, 0.0], // Error on padding should not count
                    [0.6, 0.0, 0.2, 0.0], // Error on padding should not count
                ],
                &device,
            ),
            Tensor::from_data([2, 2, 1, 1, 3, 3, 3], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(todo!(), metric.value());
    }*/
}
