use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};

/// The Top-K accuracy metric.
///
/// For K=1, this is equivalent to the [accuracy metric](`super::acc::AccuracyMetric`).
#[derive(Default)]
pub struct TopKAccuracyMetric<B: Backend> {
    k: usize,
    state: NumericMetricState,
    /// If specified, targets equal to this value will be considered padding and will not count
    /// towards the metric
    pad_token: Option<usize>,
    _b: PhantomData<B>,
}

/// The [top-k accuracy metric](TopKAccuracyMetric) input type.
#[derive(new)]
pub struct TopKAccuracyInput<B: Backend> {
    /// The outputs (batch_size, num_classes)
    outputs: Tensor<B, 2>,
    /// The labels (batch_size)
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> TopKAccuracyMetric<B> {
    /// Creates the metric.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Default::default()
        }
    }

    /// Sets the pad token.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

impl<B: Backend> Metric for TopKAccuracyMetric<B> {
    type Input = TopKAccuracyInput<B>;

    fn update(&mut self, input: &TopKAccuracyInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, _n_classes] = input.outputs.dims();

        let targets = input.targets.clone().to_device(&B::Device::default());

        let outputs = input
            .outputs
            .clone()
            .argsort_descending(1)
            .narrow(1, 0, self.k)
            .to_device(&B::Device::default())
            .reshape([batch_size, self.k]);

        let (targets, num_pad) = match self.pad_token {
            Some(pad_token) => {
                // we ignore the samples where the target is equal to the pad token
                let mask = targets.clone().equal_elem(pad_token as i64);
                let num_pad = mask.clone().int().sum().into_scalar().elem::<f64>();
                (targets.clone().mask_fill(mask, -1_i64), num_pad)
            }
            None => (targets.clone(), 0_f64),
        };

        let accuracy = targets
            .reshape([batch_size, 1])
            .repeat_dim(1, self.k)
            .equal(outputs)
            .int()
            .sum()
            .into_scalar()
            .elem::<f64>()
            / (batch_size as f64 - num_pad);

        self.state.update(
            100.0 * accuracy,
            batch_size,
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> String {
        format!("Top-K Accuracy @ TopK({})", self.k)
    }
}

impl<B: Backend> Numeric for TopKAccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_accuracy_without_padding() {
        let device = Default::default();
        let mut metric = TopKAccuracyMetric::<TestBackend>::new(2);
        let input = TopKAccuracyInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.2, 0.8], // 2, 1
                    [1.0, 2.0, 0.5], // 1, 0
                    [0.4, 0.1, 0.2], // 0, 2
                    [0.6, 0.7, 0.2], // 1, 0
                ],
                &device,
            ),
            Tensor::from_data([2, 2, 1, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(50.0, metric.value());
    }

    #[test]
    fn test_accuracy_with_padding() {
        let device = Default::default();
        let mut metric = TopKAccuracyMetric::<TestBackend>::new(2).with_pad_token(3);
        let input = TopKAccuracyInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.2, 0.8, 0.0], // 2, 1
                    [1.0, 2.0, 0.5, 0.0], // 1, 0
                    [0.4, 0.1, 0.2, 0.0], // 0, 2
                    [0.6, 0.7, 0.2, 0.0], // 1, 0
                    [0.0, 0.1, 0.2, 5.0], // Predicted padding should not count
                    [0.0, 0.1, 0.2, 0.0], // Error on padding should not count
                    [0.6, 0.0, 0.2, 0.0], // Error on padding should not count
                ],
                &device,
            ),
            Tensor::from_data([2, 2, 1, 1, 3, 3, 3], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(50.0, metric.value());
    }

    #[test]
    fn test_parameterized_unique_name() {
        let metric_a = TopKAccuracyMetric::<TestBackend>::new(2);
        let metric_b = TopKAccuracyMetric::<TestBackend>::new(1);
        let metric_c = TopKAccuracyMetric::<TestBackend>::new(2);

        assert_ne!(metric_a.name(), metric_b.name());
        assert_eq!(metric_a.name(), metric_c.name());
    }
}
